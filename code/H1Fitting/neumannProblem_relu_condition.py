"""
Neumann problem on [-1,1]^d with predetermined ReLU^k features (weak / H1 form).

PDE:  -div(α ∇u) + u = f, with Neumann boundary data g_N.
Solve for outer coefficients by assembling the H1 variational least-squares
system (mass + stiffness + Neumann boundary terms) via
minimize_linear_layer_H1_explicit_assemble_efficient_general_dim.
Inner weights are predetermined / structured (e.g. sphere sampling) with
redundant-neuron removal; derivatives of ReLU^k features are coded explicitly.

Contrast with neumann_problem_PINN.ipynb, which uses a strong-form PINN
residual + Dirichlet collocation with random ELM features.

Changelog:
  - new function: minimize_linear_layer_H1_explicit_assemble_efficient_general_dim
  - 2025 Mar 6th: general dimension d.
  - 2025 Mar 12th: fixed Monte-Carlo domain scaling bug; code working.
  - 2026 Aug 19th: added condition number computation. Added pruning of near-zero H1 neurons.
  - 2026 Aug 20th: MC sphere sampling for all dims; 5 trials; report mean ± std of κ(A).
  - 2026 Aug 20th: d=1,2 deterministic features (1 trial); d=3-6 random sphere (5 trials).
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
from scipy.sparse import linalg
from pathlib import Path
import itertools
import sympy as sp
import math  
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 
import os  

from scipy.stats import norm
from sklearn.preprocessing import normalize
from scipy.stats.qmc import Sobol

torch.set_default_dtype(torch.float64)
pi = torch.tensor(np.pi,dtype=torch.float64)
ZERO = torch.tensor([0.]).to(device)
class model(nn.Module):
    """ ReLU k shallow neural network
    Parameters: 
    input size: input dimension
    hidden_size1 : number of hidden layers 
    num_classes: output classes 
    k: degree of relu functions
    """
    def __init__(self, input_size, hidden_size1, num_classes,k = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, num_classes,bias = False)
        self.k = k 
    def forward(self, x):
        u1 = self.fc2(F.relu(self.fc1(x))**self.k)
        return u1
    def evaluate_derivative(self, x, i):
        if self.k == 1:
            u1 = self.fc2(torch.heaviside(self.fc1(x),ZERO) * self.fc1.weight.t()[i-1:i,:] )
        else:
            u1 = self.fc2(self.k*F.relu(self.fc1(x))**(self.k-1) *self.fc1.weight.t()[i-1:i,:] )  
        return u1

def plot_2D(f): 
    
    Nx = 400
    Ny = 400 
    xs = np.linspace(-1, 1, Nx)
    ys = np.linspace(-1, 1, Ny)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    xy_comb = np.stack((x.flatten(),y.flatten())).T
    xy_comb = torch.tensor(xy_comb)
    z = f(xy_comb).reshape(Nx,Ny)
    z = z.detach().numpy()
    plt.figure(dpi=200)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x , y , z )

    plt.show()

def plot_subdomains(my_model):
    x_coord =torch.linspace(0,1,200)
    wi = my_model.fc1.weight.data
    bi = my_model.fc1.bias.data 
    for i, bias in enumerate(bi):  
        if wi[i,1] !=0: 
            plt.plot(x_coord, - wi[i,0]/wi[i,1]*x_coord - bias/wi[i,1])
        else: 
            plt.plot(x_coord,  - bias/wi[i,0]*torch.ones(x_coord.size()))

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.show()
    return 0   

## Initialization
def adjust_neuron_position(my_model,target=None):
    counter = 0 
    # positions = torch.tensor([[0.,0.],[0.,1.],[1.,1.],[1.,0.]])
    positions = torch.tensor([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]])
    neuron_num = my_model.fc1.bias.size(0)
    for i in range(neuron_num): 
        w = my_model.fc1.weight.data[i:i+1,:]
        b = my_model.fc1.bias.data[i]
        values = torch.matmul(positions,w.T) # + b
        left_end = - torch.max(values)
        right_end = - torch.min(values) 
        off_set = (right_end - left_end)/1000 
        if b <= left_end + off_set: # nearly vanishing
            b = torch.rand(1)*(right_end - left_end - off_set*2) + left_end + off_set 
            my_model.fc1.bias.data[i] = b 
        if b >= right_end - off_set: # nearly nonvanishing everywhere
            if counter < 3:
                counter += 1
            else: # 3 or more 
                b = torch.rand(1)*(right_end - left_end - off_set*2) + left_end + off_set
                my_model.fc1.bias.data[i] = b 
    return my_model


def PiecewiseGQ1D_weights_points(x_l, x_r, Nx, order):
    """Piecewise Gauss quadrature weights and points on an interval."""
    x, w = np.polynomial.legendre.leggauss(order)
    gx = torch.tensor(x).to(device).view(1, -1)
    gw = torch.tensor(w).to(device).view(-1, 1)
    nodes = torch.linspace(x_l, x_r, Nx + 1).view(-1, 1).to(device)
    coef1 = (nodes[1:, :] - nodes[:-1, :]) / 2
    coef2 = (nodes[1:, :] + nodes[:-1, :]) / 2
    coef2_expand = coef2.expand(-1, gx.size(1))
    integration_points = (coef1 @ gx + coef2_expand).flatten().view(-1, 1)
    gw_expand = torch.tile(gw, (Nx, 1))
    coef1_expand = coef1.expand(coef1.size(0), gx.size(1)).flatten().view(-1, 1)
    return (coef1_expand * gw_expand).to(device), integration_points.to(device)


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
#     print("order: ",order )
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

    # print(ordered_pairs)
    # print()
    ordered_pairs = torch.tensor(ordered_pairs)
    # print(ordered_pairs.size())
    ordered_pairs = torch.tile(ordered_pairs, (1,order**2)) # number of GQ points
    # print(ordered_pairs)

    ordered_pairs =  ordered_pairs.reshape(-1,2)
    # print(ordered_pairs)
    translation = ordered_pairs*h + (torch.tensor(bl) + h/2) 
    # print(translation)

    integration_points = integration_points + translation 
#     print(integration_points.size())
    # func_values = integrand2_torch(integration_points)
    return long_weights.to(device), integration_points.to(device)


def PiecewiseGQ3D_weights_points(Nx, order,bl = [-1,-1,-1],ur = [1,1,1]): 
    """ A slight modification of PiecewiseGQ2D function that only needs the weights and integration points.
    Parameters
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

    """
    Parameters
    ----------
    target : 
        Target function 
    Nx: int 
        number of intervals along the dimension. No Ny, assume Nx = Ny
    order: int 
        order of the Gauss Quadrature
    """

    # print("order: ",order )
    x, w = np.polynomial.legendre.leggauss(order)
    gauss_pts = np.array(np.meshgrid(x,x,x,indexing='ij')).reshape(3,-1).T
    weight_list = np.array(np.meshgrid(w,w,w,indexing='ij'))
    weights =   (weight_list[0]*weight_list[1]*weight_list[2]).ravel() 

    gauss_pts =torch.tensor(gauss_pts)
    weights = torch.tensor(weights)

    # h = 1/Nx # 100 intervals 
    h = (ur[0]- bl[0])/Nx # 100 intervals 
    long_weights =  torch.tile(weights,(Nx**3,1))
    long_weights = long_weights.reshape(-1,1)
    long_weights = long_weights * h**3 /8 

    integration_points = torch.tile(gauss_pts,(Nx**3,1))
    # print("shape of integration_points", integration_points.size())
    scale_factor = h/2 
    integration_points = scale_factor * integration_points

    # index = np.arange(1,Nx+1)-0.5
    index = np.arange(0,Nx)  
    ordered_pairs = np.array(np.meshgrid(index,index,index,indexing='ij'))
    ordered_pairs = ordered_pairs.reshape(3,-1).T

    # print(ordered_pairs)
    # print()
    ordered_pairs = torch.tensor(ordered_pairs)
    # print(ordered_pairs.size())
    ordered_pairs = torch.tile(ordered_pairs, (1,order**3)) # number of GQ points
    # print(ordered_pairs)

    ordered_pairs =  ordered_pairs.reshape(-1,3)
    # print(ordered_pairs)
    # translation = ordered_pairs*h 
    translation = ordered_pairs*h + (torch.tensor(bl) + h/2) 
    # print(translation)

    integration_points = integration_points + translation 

    return long_weights.to(device), integration_points.to(device)

def MonteCarlo_Sobol_dDim_weights_points(M ,d = 4,bl = -1,ur = 1):
    
    length = ur - bl
    vol = length ** d 
    Sob_integral = torch.quasirandom.SobolEngine(dimension =d, scramble= False, seed=None) 
    integration_points = Sob_integral.draw(M).double() 
    integration_points = integration_points.to(device) * length - length/2 
    weights = torch.ones(M,1).to(device)/M * vol 
    return weights.to(device), integration_points.to(device) 


def assemble_matrix_H1_explicit_assemble_efficient_general_dim(model,target, g_N, weights, integration_points, w_bd, pts_bd, activation = 'relu',solver="direct",memory = 2**29 ):
    """ -div alpha grad u(x) + u = f 
    Parameters
    ----------
    model: 
        nn model
    alpha:
        alpha function
    target:
        rhs function f 
    pts_bd:
        integration points on the boundary, embdedded in the domain 
    """ 
    zero = torch.tensor([0.]).to(device)
    start_time = time.time() 
    w = model.fc1.weight.data 
    b = model.fc1.bias.data 
    neuron_num = b.size(0) 
    dim = integration_points.size(1) 
    M = integration_points.size(0)

    total_size = neuron_num * M # memory, number of floating numbers 
    print('total size: {} {} = {}'.format(neuron_num,M,total_size))
    num_batch = total_size//memory + 1 # divide according to memory
    print("num batches: ",num_batch)
    batch_size = M//num_batch
    start_ind = 0
    end_ind = 0 
    jac = torch.zeros(b.size(0),b.size(0)).to(device)
    rhs = torch.zeros(b.size(0),1).to(device)

    for j in range(0,M,batch_size): # batch operation in data points 
        end_ind = j + batch_size
        basis_value_col = F.relu(integration_points[j:end_ind] @ w.t()+ b)**(model.k) 
        weighted_basis_value_col = basis_value_col * weights[j:end_ind] 
        jac += weighted_basis_value_col.t() @ basis_value_col 
        rhs += weighted_basis_value_col.t() @ (target(integration_points[j:end_ind,:])) 

    # Assemble the boundary condition term <g,v>_{\Gamma_N} 
    size_pts_bd = int(pts_bd.size(0)/(2*dim))
    # M_bc = size_pts_bd 
    # total_size = M_bc * neuron_num 
    # num_batch = total_size//memory + 1 
    # batch_size = M_bc//num_batch
    if g_N != None:
        bcs_N = g_N(dim)
        for ii, g_ii in bcs_N:
            weighted_g_N = -g_ii(pts_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:])* w_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
            basis_value_bd_col = F.relu(pts_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:] @ w.t()+ b)**(model.k)
            rhs += basis_value_bd_col.t() @ weighted_g_N

            weighted_g_N = g_ii(pts_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:])* w_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:]
            basis_value_bd_col = F.relu(pts_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:] @ w.t()+ b)**(model.k)
            rhs += basis_value_bd_col.t() @ weighted_g_N
            
    # Stiffness matrix term in the jacobian 
    for d in range(dim):
        end_ind = 0 
        if model.k == 1:  
            for j in range(0,M,batch_size): 
                end_ind = j + batch_size 
                basis_value_dxi_col = torch.heaviside(integration_points[j:end_ind] @ w.t()+ b, zero) * w.t()[d:d+1,:]
                weighted_basis_value_dx_col = basis_value_dxi_col * weights[j:end_ind] 
                jac += weighted_basis_value_dx_col.t() @ basis_value_dxi_col 
#             basis_value_dxi_col = torch.heaviside(integration_points @ w.t()+ b, zero) * w.t()[d:d+1,:]
#             weighted_basis_value_dx_col = basis_value_dxi_col * weights * coef_alpha 
#             jac += weighted_basis_value_dx_col.t() @ basis_value_dxi_col 

        else: 
            for j in range(0,M,batch_size):  
                end_ind = j + batch_size 
                basis_value_dxi_col = model.k * F.relu(integration_points[j:end_ind] @ w.t()+ b)**(model.k-1) * w.t()[d:d+1,:]
                weighted_basis_value_dx_col = basis_value_dxi_col * weights[j:end_ind]  
                jac += weighted_basis_value_dx_col.t() @ basis_value_dxi_col 
#             basis_value_dxi_col = model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[d:d+1,:]
#             weighted_basis_value_dx_col = basis_value_dxi_col * weights * coef_alpha  
#             jac += weighted_basis_value_dx_col.t() @ basis_value_dxi_col 

    print("assembling the mass matrix time taken: ", time.time()-start_time) 

#     start_time = time.time()    
#     if solver == "cg": 
#         sol, exit_code = linalg.cg(np.array(jac.detach().cpu()),np.array(rhs.detach().cpu()),tol=1e-12)
#         sol = torch.tensor(sol).view(1,-1)
#     elif solver == "direct": 
# #         sol = np.linalg.inv( np.array(jac.detach().cpu()) )@np.array(rhs.detach().cpu())
#         sol = (torch.linalg.solve( jac.detach(), rhs.detach())).view(1,-1)
#     elif solver == "ls":
#         sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd').solution).view(1,-1)
#         # sol = (torch.linalg.lstsq(jac.detach(),rhs.detach()).solution).view(1,-1) # gpu/cpu, driver = 'gels', cannot solve singular
#     print("solving Ax = b time taken: ", time.time()-start_time)
    return jac.detach().cpu(),rhs.detach().cpu() 


def initialize_model_1d(my_model):
    # Uniform grid on the upper semicircle of S^1: (ω, b) = (cos θ, sin θ), θ ∈ [0, π).
    neuron_nums = my_model.fc1.bias.size(0)
    theta = torch.linspace(0, pi - 0.001, neuron_nums + 1)[:-1]
    W = torch.cos(theta).view(-1, 1)
    b = torch.sin(theta)
    my_model.fc1.weight.data[:, :] = W[:, :]
    my_model.fc1.bias.data[:] = b[:]
    return my_model


def initialize_model_1(my_model):
    # w ~ U(S^1), b ~ U(-1.42,1.42) 
    neuron_nums = my_model.fc1.bias.size(0)
    samples = torch.rand(neuron_nums,2) 
    T =torch.tensor([[2*pi,0],[0,2.84]])
    shift = torch.tensor([0,-1.42]) 
    samples = samples@T + shift 
    theta = samples[:,0].reshape(neuron_nums,1)
    W1 = torch.cos(theta)
    W2 = torch.sin(theta)
    W = torch.cat((W1,W2),1) # N1 x 2
    b = samples[:,1].reshape(neuron_nums,1)
    my_model.fc1.weight.data[:,:] = W[:,:]
    my_model.fc1.bias.data[:] = b[:,0] 
    
    return my_model 

def initialize_model_2(my_model,dim = 2):
    # (w,b) ~ U(S^2)
    neuron_nums = my_model.fc1.bias.size(0)
    points = torch.randn(neuron_nums,dim + 1)
    points = points/torch.norm(points, dim=1, keepdim=True)
    my_model.fc1.weight.data[:,:] = points[:,0:dim]
    my_model.fc1.bias.data[:] = points[:,dim]  
    return my_model 


def initialize_model_2_qmc(my_model,dim = 2):
    # (w,b) ~ U(S^2)
    neuron_nums = my_model.fc1.bias.size(0)
    sobol_engine = Sobol(dim+1,scramble=False)
    u = sobol_engine.random(n=neuron_nums)
    
    # points = torch.randn(neuron_nums,dim + 1)
    epsilon = np.finfo(float).eps
    u = np.clip(u, epsilon, 1 - epsilon)

    # Inverse CDF to get standard normal points
    z = norm.ppf(u)

    # Normalize to project onto sphere
    x = normalize(z, axis=1)
    # Convert to torch tensor
    points = torch.tensor(x, dtype=torch.float64)
    points = points/torch.norm(points, dim=1, keepdim=True)
    my_model.fc1.weight.data[:,:] = points[:,0:dim]
    my_model.fc1.bias.data[:] = points[:,dim]  
    return my_model 

def initialize_model_3(my_model):
    # generate a uniform grid on S^2 
    neuron_nums = my_model.fc1.bias.size(0) 

    indices = torch.arange(0, neuron_nums, dtype=torch.float) + 0.5
    phi = torch.acos(1 - 2*indices/neuron_nums)
    theta = pi * (1 + 5**0.5) * indices
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    points = torch.stack((x, y, z), dim=1)
    my_model.fc1.weight.data[:,:] = points[:,0:2]
    my_model.fc1.bias.data[:] = points[:,2]
    return my_model 

def remove_redundant_neuron(my_model, dims = 3, choice = 2): 
    ##  choice 1:  [0,1]^d, choice 2: [-1,1]^d
    def create_mesh_grid(dims, pts):
        mesh = torch.tensor(list(itertools.product(pts,repeat=dims)))
        vertices = mesh.reshape(len(pts) ** dims, -1) 
        return vertices
    counter = 0 
    # positions = torch.tensor([[0.,0.],[0.,1.],[1.,1.],[1.,0.]])
    # pts = torch.tensor([0.,1.]) # for domain [0,1]^d 
    if choice == 1: 
        pts = torch.tensor([0.,1.])
    elif choice == 2:
        pts = torch.tensor([-1.,1.])# for domain [-1,1]^d 
    elif choice == 3:
        pts = torch.tensor([-1./2,1./2])# for domain [-1,1]^d 
    positions = create_mesh_grid(dims,pts) 
    neuron_num = my_model.fc1.bias.size(0)
    relu_k = my_model.k 
    recorded_neurons = []
    poly_dofs = math.comb(relu_k + dims, dims)
    for i in range(neuron_num): 
        w = my_model.fc1.weight.data[i:i+1,:]
        b = my_model.fc1.bias.data[i]
        values = torch.matmul(positions,w.T)
        left_end = - torch.max(values)
        right_end = - torch.min(values)
        offset = (right_end - left_end)/50
        if b > left_end + offset/2 and b < right_end - offset/2: 
            recorded_neurons.append((w, b))
        elif b >= right_end - offset/2 and counter < poly_dofs:
            recorded_neurons.append((w, b))
            counter += 1

    new_neuron_num = len(recorded_neurons)
    new_model = model(dims, new_neuron_num, 1, k=relu_k).to(device)
    for i, (w, b) in enumerate(recorded_neurons):
        new_model.fc1.weight.data[i:i+1,:] = w
        new_model.fc1.bias.data[i] = b
    print("Number of neurons removed: ", neuron_num - new_neuron_num)
    print("Number of neurons left: ", new_neuron_num)  
    return new_model


def prune_near_zero_h1_neurons(A, rel_tol=1e-6):
    """Drop neurons whose H1 energy A_ii is tiny compared with the largest neuron.

    Vertex-based remove_redundant_neuron can miss a hyperplane that only clips a
    thin sliver of [-1,1]^d (vertices still sit on both sides). For ReLU^k, k>=2,
    that sliver has near-zero H1 mass and makes the Gram matrix numerically singular.
    rel_tol=1e-6 also drops weakly activated neurons that still inflate cond(A).

    Returns (n_keep, A_reduced, n_drop). Does not allocate a new nn.Module, so the
    torch RNG used by later initializations is left unchanged.
    """
    A_np = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A, dtype=np.float64)
    diag = np.diag(A_np)
    keep = diag >= rel_tol * np.max(diag)
    n_keep = int(np.count_nonzero(keep))
    n_drop = int(np.size(keep) - n_keep)
    if n_drop == 0:
        return n_keep, A_np, 0
    print("Near-zero H1 neurons removed: ", n_drop)
    print("Number of neurons left: ", n_keep)
    return n_keep, A_np[np.ix_(keep, keep)], n_drop


def galerkin_condition_number(A):
    A_np = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A, dtype=np.float64)
    return float(np.linalg.cond(A_np))


folder = 'condition_number_results/'
os.makedirs(folder, exist_ok=True)
log_path = os.path.join(folder, 'progress.log')
_run_t0 = time.time()

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {time.time()-_run_t0:7.1f}s  {msg}"
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')

with open(log_path, 'w') as f:
    f.write('')
log(f"start  device={device}")

relu_k = 3
n_trials = 5
mc_seeds = list(range(n_trials))  # seeds 0..4


def make_target(d):
    def target(x):
        return (d * (pi / 2) ** 2 + 1) * torch.prod(torch.sin(pi / 2 * x), dim=1, keepdim=True)
    return target


def quadrature_for_dim(d):
    if d == 1:
        wts, pts = PiecewiseGQ1D_weights_points(-1, 1, 1024, 5)
    elif d == 2:
        wts, pts = PiecewiseGQ2D_weights_points(100, 3, [-1, -1], [1, 1])
    elif d == 3:
        wts, pts = PiecewiseGQ3D_weights_points(50, 3, [-1, -1, -1], [1, 1, 1])
    elif d == 4:
        wts, pts = MonteCarlo_Sobol_dDim_weights_points(int(1e6), d=d, bl=-1, ur=1)
    elif d == 5:
        wts, pts = MonteCarlo_Sobol_dDim_weights_points(int(6e6), d=d, bl=-1, ur=1)
    elif d == 6:
        wts, pts = MonteCarlo_Sobol_dDim_weights_points(int(5e7), d=d, bl=-1, ur=1)
    else:
        raise ValueError(f"unsupported d={d}")
    return wts, pts


def neuron_counts_for_dim(d):
    # 1D-3D: 8..512; 4D-6D: 8..256
    if d <= 3:
        return np.array([2 ** i for i in range(3, 10)])
    return np.array([2 ** i for i in range(3, 9)])


def initialize_features(my_model, d):
    """Deterministic dictionaries for d=1,2; random sphere for d>=3."""
    if d == 1:
        return initialize_model_1d(my_model.cpu()).to(device)
    if d == 2:
        return initialize_model_3(my_model).to(device)
    return initialize_model_2(my_model, dim=d).to(device)


def run_condition_for_dim(d):
    """d=1,2: one deterministic trial; d>=3: n_trials random sphere seeds."""
    neuron_num_arr = neuron_counts_for_dim(d)
    n_sizes = len(neuron_num_arr)
    integration_weights, integration_points = quadrature_for_dim(d)
    target = make_target(d)
    mem = 2 ** 27 if d >= 3 else 2 ** 29
    use_random = d >= 3
    seeds = mc_seeds if use_random else [None]

    cond_trials = np.full((len(seeds), n_sizes), np.nan)
    n_kept_trials = np.full((len(seeds), n_sizes), np.nan)
    tiny_trials = np.full((len(seeds), n_sizes), np.nan)

    log(
        f"{d}D begin  M={integration_points.size(0)}  N={list(neuron_num_arr)}  "
        f"trials={len(seeds)}  features={'random' if use_random else 'deterministic'}"
    )
    for t, seed in enumerate(seeds):
        if seed is not None:
            torch.manual_seed(seed)
        for j, neuron_num in enumerate(neuron_num_arr):
            my_model = model(d, neuron_num, 1, relu_k).to(device)
            my_model = initialize_features(my_model, d)
            my_model = remove_redundant_neuron(my_model.cpu(), dims=d, choice=2).to(device)
            A, _ = assemble_matrix_H1_explicit_assemble_efficient_general_dim(
                my_model, target, None, integration_weights, integration_points,
                torch.tensor([]), torch.tensor([]),
                activation='relu', solver='direct', memory=mem,
            )
            n_keep, A, n_tiny = prune_near_zero_h1_neurons(A)
            cond = galerkin_condition_number(A)
            cond_trials[t, j] = cond
            n_kept_trials[t, j] = n_keep
            tiny_trials[t, j] = n_tiny
            seed_str = f"seed={seed}" if seed is not None else "det"
            log(f"{d}D  {seed_str}  N={neuron_num} -> {n_keep}  tinyH1={n_tiny}  cond={cond:.3e}")

    cond_mean = np.nanmean(cond_trials, axis=0)
    if cond_trials.shape[0] > 1:
        cond_std = np.nanstd(cond_trials, axis=0, ddof=1)
        n_kept_std = np.nanstd(n_kept_trials, axis=0, ddof=1)
    else:
        cond_std = np.zeros_like(cond_mean)
        n_kept_std = np.zeros_like(cond_mean)
    n_kept_mean = np.nanmean(n_kept_trials, axis=0)

    out_path = f'{folder}Neumann-problem-Predetermined-Feature-{d}d-relu{relu_k}.npz'
    np.savez(
        out_path,
        neuron_num_arr=neuron_num_arr,
        seeds=np.array([s if s is not None else -1 for s in seeds]),
        condition_number_trials=cond_trials,
        condition_number_arr=cond_mean,
        condition_number_mean=cond_mean,
        condition_number_std=cond_std,
        actual_neuron_trials=n_kept_trials,
        actual_neuron_arr=n_kept_mean,
        actual_neuron_mean=n_kept_mean,
        actual_neuron_std=n_kept_std,
        tiny_h1_trials=tiny_trials,
    )
    for j, N in enumerate(neuron_num_arr):
        log(
            f"{d}D  mean  N={N} -> {n_kept_mean[j]:.1f}+/-{n_kept_std[j]:.1f}  "
            f"cond={cond_mean[j]:.3e}+/-{cond_std[j]:.3e}"
        )
    log(f"{d}D saved {out_path}")


for d in range(1, 7):
    run_condition_for_dim(d)
log("all done")
 
