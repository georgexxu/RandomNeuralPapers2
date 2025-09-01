import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import itertools
import scipy 
import math
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
pi = torch.tensor(np.pi,dtype=torch.float64)

def split_domain_boundary(pts, tol=1e-10):
    """
    Split points in [0,1]^2 into interior (domain) and boundary sets.

    Args:
        pts: (N,2) tensor, points in [0,1]^2
        tol: float, tolerance for boundary check

    Returns:
        pts_domain: (Nd,2) tensor, interior points
        pts_boundary: (Nb,2) tensor, boundary points
    """
    x, y = pts[:, 0], pts[:, 1]

    # mask for boundary: if near 0 or 1
    mask_boundary = (
        (x < tol) | (x > 1 - tol) |
        (y < tol) | (y > 1 - tol)
    )

    pts_boundary = pts[mask_boundary]
    pts_domain = pts[~mask_boundary]

    return pts_domain, pts_boundary

def SquareMesh2D_points(xmin=-1, xmax=1, Nx=150):
    x = torch.linspace(xmin, xmax,Nx)
    X1, X2 = torch.meshgrid(x,x, indexing='ij')
    x_test = torch.concat([X1.reshape(-1,1), X2.reshape(-1,1)], dim=1)
    return x_test

def PiecewiseGQ2D_weights_points(Nx, order,bl = [-1,-1],ur = [1,1]): 
    """ A slight modification of PiecewiseGQ2D function that only needs the weights and integration points.
    Parameters
    Allows a symmetric square domain (around 0) with lower left corner at bl and upper right corner at ur 
    ----------
    Nx: int 
        number of intervals along the dimension. No Ny, assume Nx = Ny
    order: int 
        order of the Gauss Quadrature
    Returns
    -------
    long_weights: torch.tensor
    integration_points: torch.tensor
    """
    x, w = np.polynomial.legendre.leggauss(order)
    gauss_pts = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T
    weights =  (w*w[:,None]).ravel()

    gauss_pts =torch.tensor(gauss_pts)
    weights = torch.tensor(weights)

    h = (ur[0]- bl[0])/Nx # 100 intervals 
    long_weights =  torch.tile(weights,(Nx**2,1))
    long_weights = long_weights.reshape(-1,1)
    long_weights = long_weights * h**2 /4 

    integration_points = torch.tile(gauss_pts,(Nx**2,1))
    scale_factor = h/2 
    integration_points = scale_factor * integration_points

    index = np.arange(0,Nx)  
    ordered_pairs = np.array(np.meshgrid(index,index,indexing='ij'))
    ordered_pairs = ordered_pairs.reshape(2,-1).T

    ordered_pairs = torch.tensor(ordered_pairs)
    ordered_pairs = torch.tile(ordered_pairs, (1,order**2)) # number of GQ points

    ordered_pairs =  ordered_pairs.reshape(-1,2)
    translation = ordered_pairs*h + (torch.tensor(bl) + h/2) 

    integration_points = integration_points + translation 
    return long_weights, integration_points

def s2_uniform_grid(neuron_nums):
    indices = torch.arange(0, neuron_nums, dtype=torch.float64) + 0.5
    phi = torch.acos(1 - 2*indices/neuron_nums)
    theta = pi * (1 + 5**0.5) * indices
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    points = torch.stack((x, y, z), dim=1)
    return points 

def s2_uniform_grid_init(neuron_num, dims=2, k=1):
    points = s2_uniform_grid(neuron_num)
    def create_mesh_grid(dims, pts):
        mesh = torch.tensor(list(itertools.product(pts,repeat=dims)))
        vertices = mesh.reshape(len(pts) ** dims, -1) 
        return vertices
    
    counter = 0 
    pts = torch.tensor([0.,1.])  
    # pts = torch.tensor([-1.,1.])
    positions = create_mesh_grid(dims,pts) 
    neuron_num = points.shape[0]
    relu_k = k
    ws = []
    bs = []
    poly_dofs = math.comb(relu_k + dims, dims)
    for i in range(neuron_num): 
        w = points[i:i+1,0:2]
        b = points[i,2]
        values = torch.matmul(positions,w.T)
        left_end = - torch.max(values)
        right_end = - torch.min(values)
        offset = (right_end - left_end)/50
        if b > left_end + offset/2 and b < right_end - offset/2: 
            ws.append(w)
            bs.append(b)
        elif b >= right_end - offset/2 and counter < poly_dofs:
            ws.append(w)
            bs.append(b)
            counter += 1
    
    ws = torch.concatenate(ws)
    bs = torch.tensor(bs)
    return ws, bs

def uniform_rand_init(neuron_num, dims=2, s=1):
    ws = (torch.rand(neuron_num, dims,dtype=torch.float64) - 0.5) * s
    bs = (torch.rand(neuron_num,dtype=torch.float64) - 0.5) * s
    return ws, bs

# show convergence order 
def output_convergence_order_l2(neuron_nums,err_list_l2): 
    print("$n$ & \t $\|u-u_n \|_{L^2}$ & \t order  \\\ \hline \hline ")
    for i, item in enumerate(err_list_l2):
        if i == 0: 
            print("{} \t\t & {:.3e} &\t\t *  \\\ \hline  \n".format(neuron_nums[i],item))    
        else: 
            print("{} \t\t &  {:.3e} &  \t\t {:.2f}  \\\ \hline  \n".format(neuron_nums[i],item, np.log(err_list_l2[i-1]/err_list_l2[i])/np.log(neuron_nums[i]/neuron_nums[i-1]) ) )

def plot_err_convergence(relu_k,d,actual_neuron_arr,mean_err_l2_arr,title): 
    optimal_rate = 1/2 + (2 * relu_k +1)/(2*d)
    ref = mean_err_l2_arr[0] *actual_neuron_arr[0]**optimal_rate * (actual_neuron_arr)**(-optimal_rate)
    plt.plot(actual_neuron_arr,ref,'-.',label = 'slope =  -{}'.format(optimal_rate))
    plt.plot(actual_neuron_arr, mean_err_l2_arr, '.-', label = 'ReLU$^{}$'.format(relu_k), linewidth = 2)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylabel("rel $L^2$ error")
    plt.xlabel("Number of neurons n")
    plt.title(title)
    plt.show() 

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

def plot_err_convergence_levels(relu_k, d, neuron_dict, err_dict, levels, title, outpath):
    """
    Plot L2 error convergence for multiple levels l, saving to file only.

    Parameters:
        relu_k : int
            Degree of ReLU (k)
        d : int
            Input dimension
        neuron_dict : dict of arrays
            Each element: actual_neuron_arr for one level l
        err_dict : dict of arrays
            Each element: mean_err_l2_arr for one level l
        levels : list
            List of level parameters corresponding to neuron_dict / err_dict
        title : str
            Plot title
        outpath : str
            Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    L2J = {1: 4, 2: 9, 3: 25}

    for l in levels:
        neurons = neuron_dict[l]
        err = err_dict[l]
        if l == 0:
            optimal_rate = 1/2 + (2 * relu_k + 1) / (2 * d)
                        
            if ("Delta" in title) & ("4, 4" in title):
                scale = 10**4
            elif "Delta" in title:
                scale = 10**2
            else:
                scale = 1
            ref = err[0] * neurons[0]**optimal_rate * neurons**(-optimal_rate) * scale 
            ax.plot(neurons, ref, '--', label=f'slope = -{optimal_rate:.2f}')
            ax.plot(neurons, err, '.-', label=f'ReLU$^{relu_k}$', linewidth=2)
        else:
            ax.plot(neurons, err, '.-', label=f'J = {L2J[l]}, ReLU$^{relu_k}$', linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of neurons n")
    ax.set_ylabel("rel $L^2$ error")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tanh_err_convergence_levels(scale, d, neuron_dict, err_dict, levels, title, outpath):
    """
    Plot L2 error convergence for multiple levels l, saving to file only.

    Parameters:
        scale : int
            Scale of uniform distribution
        d : int
            Input dimension
        neuron_dict : dict of arrays
            Each element: actual_neuron_arr for one level l
        err_dict : dict of arrays
            Each element: mean_err_l2_arr for one level l
        levels : list
            List of level parameters corresponding to neuron_dict / err_dict
        title : str
            Plot title
        outpath : str
            Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    L2J = {1: 4, 2: 9, 3: 25}

    for l in levels:
        neurons = neuron_dict[l]
        err = err_dict[l]
        if l == 0:
            ax.plot(neurons, err, '.-', label=f'Tanh-${scale}$', linewidth=2)
        else:
            ax.plot(neurons, err, '.-', label=f'J = {L2J[l]}, Tanh-${scale}$', linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of neurons n")
    ax.set_ylabel("rel $L^2$ error")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
