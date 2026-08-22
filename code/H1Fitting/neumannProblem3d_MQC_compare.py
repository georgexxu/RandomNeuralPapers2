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


def minimize_linear_layer_H1_explicit_assemble_efficient_general_dim(model,target, g_N, weights, integration_points, w_bd, pts_bd, activation = 'relu',solver="direct",memory = 2**29 ):
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

    start_time = time.time()    
    if solver == "cg": 
        sol, exit_code = linalg.cg(np.array(jac.detach().cpu()),np.array(rhs.detach().cpu()),tol=1e-12)
        sol = torch.tensor(sol).view(1,-1)
    elif solver == "direct": 
#         sol = np.linalg.inv( np.array(jac.detach().cpu()) )@np.array(rhs.detach().cpu())
        sol = (torch.linalg.solve( jac.detach(), rhs.detach())).view(1,-1)
    elif solver == "ls":
        sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd').solution).view(1,-1)
        # sol = (torch.linalg.lstsq(jac.detach(),rhs.detach()).solution).view(1,-1) # gpu/cpu, driver = 'gels', cannot solve singular
    print("solving Ax = b time taken: ", time.time()-start_time)
    return sol 

def minimize_linear_layer_H1_explicit_assemble_efficient(model,target,weights, integration_points,activation = 'relu',solver="direct" ):

    # weights, integration_points = PiecewiseGQ2D_weights_points(Nx, order) 
    # integration_points.requires_grad_(True) 
    start_time = time.time() 
    w = model.fc1.weight.data 
    b = model.fc1.bias.data 
    neuron_num = b.size(0) 

    if activation == 'relu':
        basis_value_col = F.relu(integration_points @ w.t()+ b)**(model.k) 
        if model.k == 1:  
            basis_value_dx_col = torch.heaviside(integration_points @ w.t()+ b, torch.tensor([0.])) * w.t()[0:1,:] 
            basis_value_dy_col = torch.heaviside(integration_points @ w.t()+ b,torch.tensor([0.])) * w.t()[1:2,:] 
        else: 
            basis_value_dx_col = model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[0:1,:]
            basis_value_dy_col = model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[1:2,:] 
    # elif activation == 'tanh': 
    #     basis_value_col = torch.tanh(integration_points @ w.t()+ b) 
    #     basis_value_dx_col = tanh_activation_dx(integration_points @ w.t()+ b) * w.t()[0:1,:]
    #     basis_value_dy_col = tanh_activation_dx(integration_points @ w.t()+ b) * w.t()[1:2,:]
    # elif activation == 'gaussian':
    #     basis_value_col = Gaussian_activation(integration_points @ w.t()+ b)
    #     basis_value_dx_col = Gaussian_activation_dx(integration_points @ w.t()+ b) * w.t()[0:1,:]
    #     basis_value_dy_col = Gaussian_activation_dx(integration_points @ w.t()+ b) * w.t()[1:2,:]
    # elif activation == 'cosine':
    #     basis_value_col = cosine_activation(integration_points @ w.t()+ b) 
    #     basis_value_dx_col = cosine_activation_dx(integration_points @ w.t()+ b) * w.t()[0:1,:]
    #     basis_value_dy_col = cosine_activation_dx(integration_points @ w.t()+ b) * w.t()[1:2,:] 

    weighted_basis_value_col = basis_value_col * weights 
    jac1 = weighted_basis_value_col.t() @ basis_value_col  # mass matrix 
    rhs = weighted_basis_value_col.t() @ (target(integration_points)) 
    print("assembling the mass matrix time taken: ", time.time()-start_time) 

    start_time = time.time() 
    weighted_basis_value_dx_col = basis_value_dx_col * weights
    weighted_basis_value_dy_col = basis_value_dy_col * weights
    jac2 = weighted_basis_value_dx_col.t() @ basis_value_dx_col + weighted_basis_value_dy_col.t() @ basis_value_dy_col 
    print("assembling the stiffness matrix time taken: ", time.time()-start_time)   
    jac = jac1 + jac2    
    
    start_time = time.time()    
    if solver == "cg": 
        sol, exit_code = linalg.cg(np.array(jac.detach().cpu()),np.array(rhs.detach().cpu()),tol=1e-12)
        sol = torch.tensor(sol).view(1,-1)
    elif solver == "direct": 
#         sol = np.linalg.inv( np.array(jac.detach().cpu()) )@np.array(rhs.detach().cpu())
        sol = (torch.linalg.solve( jac.detach(), rhs.detach())).view(1,-1)
    elif solver == "ls":
        sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd').solution).view(1,-1)
        # sol = (torch.linalg.lstsq(jac.detach(),rhs.detach()).solution).view(1,-1) # gpu/cpu, driver = 'gels', cannot solve singular
    print("solving Ax = b time taken: ", time.time()-start_time)
    return sol 

def show_convergence_order(err_l2,err_h10,exponent,dict_size, filename,write2file = False):
    
    if write2file:
        file_mode = "a" if os.path.exists(filename) else "w"
        f_write = open(filename, file_mode)
    
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    err_list2 = [err_h10[i] for i in neuron_nums ] 
    # f_write.write('M:{}, relu {} \n'.format(M,k))
    if write2file:
        f_write.write('dictionary size: {}\n'.format(dict_size))
        f_write.write("neuron num \t\t error \t\t order \t\t h10 error \\ order \n")
    print("neuron num \t\t error \t\t order")
    for i, item in enumerate(err_list):
        if i == 0: 
            # print(neuron_nums[i], end = "\t\t")
            # print(item, end = "\t\t")
            
            # print("*")
            print("{} \t\t {:.6f} \t\t * \t\t {:.6f} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t * \t\t {} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ))
        else: 
            # print(neuron_nums[i], end = "\t\t")
            # print(item, end = "\t\t") 
            # print(np.log(err_list[i-1]/err_list[i])/np.log(2))
            print("{} \t\t {:.6f} \t\t {:.6f} \t\t {:.6f} \t\t {:.6f} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t {} \t\t {} \t\t {} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ))
    if write2file:     
        f_write.write("\n")
        f_write.close()

def show_convergence_order_latex(err_l2,err_h10,exponent): 
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    err_list2 = [err_h10[i] for i in neuron_nums ] 
    print("neuron num  & \t $\|u-u_n \|_{L^2}$ & \t order & \t $ | u -u_n |_{H^1}$ & \t order \\\ \hline \hline ")
    for i, item in enumerate(err_list):
        if i == 0: 
            print("{} \t\t & {:.6f} &\t\t * & \t\t {:.6f} & \t\t *  \\\ \hline  \n".format(neuron_nums[i],item, err_list2[i] ) )   
        else: 
            print("{} \t\t &  {:.3e} &  \t\t {:.2f} &  \t\t {:.3e} &  \t\t {:.2f} \\\ \hline  \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ) )

## helper functions 

# show convergence order 
def output_convergence_order(neuron_nums,err_list_l2, err_list_h2): 
    print("$n$ & \t $\|u-u_n \|_{L^2}$ & \t order  & $ |u-u_n |_{H^1}$ & 	 order\\\ \hline \hline ")
    for i, item in enumerate(err_list_l2):
        if i == 0: 
            print("{} \t\t & {:.3e} &\t\t * & {:.3e} &  *  \\\ \hline  \n".format(neuron_nums[i],item, err_list_h2[i]) )   
        else: 
            print("{} \t\t &  {:.3e} &  \t\t {:.2f}  &  {:.3e} &  {:.2f}  \\\ \hline  \n".format(neuron_nums[i],item, np.log(err_list_l2[i-1]/err_list_l2[i])/np.log(neuron_nums[i]/neuron_nums[i-1]), err_list_h2[i], np.log(err_list_h2[i-1]/err_list_h2[i])/np.log(neuron_nums[i]/neuron_nums[i-1]) ) )


def compute_l2_error(u_exact,my_model,M,batch_size_2,weights,integration_points): 
    err = 0 
    if my_model == None: 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            func_values = u_exact(integration_points[jj:end_index,:])
            err += torch.sum(func_values**2 * weights[jj:end_index,:])
    else: 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            func_values = u_exact(integration_points[jj:end_index,:]) - my_model(integration_points[jj:end_index,:]).detach()
            err += torch.sum(func_values**2 * weights[jj:end_index,:])
    return err**0.5 

def compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,weights,integration_points):
    """
    Parameters
    ----------
    u_exact_grad: list or None
        a list that contains ways of evaluating partial derivatives that gives the gradient  
    """
    err_h10 = 0 
     # initial gradient error 
    if u_exact_grad != None and my_model!=None:
        u_grad = u_exact_grad() 
        for ii, grad_i in enumerate(u_grad): 
            for jj in range(0,M,batch_size_2): 
                end_index = jj + batch_size_2 
                my_model_dxi = my_model.evaluate_derivative(integration_points[jj:end_index,:],ii+1).detach() 
                err_h10 += torch.sum((grad_i(integration_points[jj:end_index,:]) - my_model_dxi)**2 * weights[jj:end_index,:])
    elif u_exact_grad != None and my_model==None:
        u_grad = u_exact_grad() 
        for grad_i in u_grad: 
            for jj in range(0,M,batch_size_2): 
                end_index = jj + batch_size_2 
                err_h10 += torch.sum((grad_i(integration_points[jj:end_index,:]))**2 * weights[jj:end_index,:])
    return err_h10**0.5

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


##1. exact solution: sin(pi/2 x)sin(pi/2 y) 
def u_exact(x): 
    z = torch.sin(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3])  
    return z 
def u_x(x):
    z = (pi/2)*torch.cos(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3])  
    return z
def u_y(x):
    z = (pi/2)*torch.sin(pi/2*x[:,0:1])*torch.cos(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3])  
    return z 
def u_z(x):
    z = (pi/2)*torch.sin(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.cos(pi/2*x[:,2:3])  
    return z 
def target(x):
    z = (3 * (pi/2)**2 + 1)*torch.sin(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3]) 
    return z
def u_exact_grad(): 
    return [u_x, u_y, u_z]

Nx = 50   
order = 3 
integration_weights, integration_points = PiecewiseGQ3D_weights_points(Nx, order,[-1,-1,-1],[1,1,1])
M = integration_points.size(0)
neuron_num_list = [25,50,100,200,400,800,1600]
actual_neuron_list = []
err_list_l2 = []
err_list_h1 = []
relu_k = 3 
for neuron_num in neuron_num_list: 
    my_model = model(3,neuron_num,1,relu_k).to(device)
    my_model = initialize_model_2(my_model,dim=3).to(device)
    my_model = remove_redundant_neuron(my_model.cpu(), dims = 3, choice = 2).to(device) 
    solver = 'direct'
    # sol = minimize_linear_layer_H1_explicit_assemble_efficient(my_model,target,integration_weights, integration_points,activation = 'relu',solver = solver)
    sol = minimize_linear_layer_H1_explicit_assemble_efficient_general_dim(my_model,target,None,integration_weights, integration_points,torch.tensor([]),torch.tensor([]),activation = 'relu',solver = solver)
    my_model.fc2.weight.data[0,:] = sol[:] 
    actual_neuron_list.append(my_model.fc1.bias.size(0)) 
    # compute the error 
    memory = 2**28 
    num_neuron = 0 if my_model == None else int(my_model.fc1.bias.detach().data.size(0))
    total_size2 = M*(num_neuron+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # in
    
#     errl2 = (integration_weights.t()@(u_exact(integration_points) - my_model(integration_points).detach())**2)**0.5
    errl2 = compute_l2_error(u_exact,my_model,M,batch_size_2,integration_weights,integration_points)
#     errh1 = integration_weights.t()@(u_x(integration_points) - my_model.evaluate_derivative(integration_points,1).detach())**2
#     errh1 += integration_weights.t()@(u_y(integration_points) - my_model.evaluate_derivative(integration_points,2).detach())**2
#     errh1 += integration_weights.t()@(u_z(integration_points) - my_model.evaluate_derivative(integration_points,3).detach())**2 
#     errh1 = errh1**0.5    
    errh1 = compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,integration_weights,integration_points)

    # print("L2 error: ",errl2)
    # print("H1 error: ",errh1) 
    err_list_l2.append(errl2.item())
    err_list_h1.append(errh1.item())
    # plot_2D(my_model.cpu())  

d = 3 
print("L2 order:",0.5 + (2*relu_k +1)/(2*d))
print("H1 order:",0.5 + (2*(relu_k-1) +1)/(2*d))

output_convergence_order(actual_neuron_list,err_list_l2, err_list_h1)


##1. exact solution: sin(pi/2 x)sin(pi/2 y) 
def u_exact(x): 
    z = torch.sin(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3])  
    return z 
def u_x(x):
    z = (pi/2)*torch.cos(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3])  
    return z
def u_y(x):
    z = (pi/2)*torch.sin(pi/2*x[:,0:1])*torch.cos(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3])  
    return z 
def u_z(x):
    z = (pi/2)*torch.sin(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.cos(pi/2*x[:,2:3])  
    return z 
def target(x):
    z = (3 * (pi/2)**2 + 1)*torch.sin(pi/2*x[:,0:1])*torch.sin(pi/2*x[:,1:2] ) * torch.sin(pi/2*x[:,2:3]) 
    return z
def u_exact_grad(): 
    return [u_x, u_y, u_z]

Nx = 50   
order = 3 
integration_weights, integration_points = PiecewiseGQ3D_weights_points(Nx, order,[-1,-1,-1],[1,1,1])
M = integration_points.size(0)
neuron_num_list = [25,50,100,200,400,800,1600]
actual_neuron_list = []
err_list_l2 = []
err_list_h1 = []
relu_k = 3 
for neuron_num in neuron_num_list: 
    my_model = model(3,neuron_num,1,relu_k).to(device)
    my_model = initialize_model_2_qmc(my_model,dim=3).to(device)
    my_model = remove_redundant_neuron(my_model.cpu(), dims = 3, choice = 2).to(device) 
    solver = 'direct'
    # sol = minimize_linear_layer_H1_explicit_assemble_efficient(my_model,target,integration_weights, integration_points,activation = 'relu',solver = solver)
    sol = minimize_linear_layer_H1_explicit_assemble_efficient_general_dim(my_model,target,None,integration_weights, integration_points,torch.tensor([]),torch.tensor([]),activation = 'relu',solver = solver)
    my_model.fc2.weight.data[0,:] = sol[:] 
    actual_neuron_list.append(my_model.fc1.bias.size(0)) 
    # compute the error 
    memory = 2**28 
    num_neuron = 0 if my_model == None else int(my_model.fc1.bias.detach().data.size(0))
    total_size2 = M*(num_neuron+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # in
    
#     errl2 = (integration_weights.t()@(u_exact(integration_points) - my_model(integration_points).detach())**2)**0.5
    errl2 = compute_l2_error(u_exact,my_model,M,batch_size_2,integration_weights,integration_points)
#     errh1 = integration_weights.t()@(u_x(integration_points) - my_model.evaluate_derivative(integration_points,1).detach())**2
#     errh1 += integration_weights.t()@(u_y(integration_points) - my_model.evaluate_derivative(integration_points,2).detach())**2
#     errh1 += integration_weights.t()@(u_z(integration_points) - my_model.evaluate_derivative(integration_points,3).detach())**2 
#     errh1 = errh1**0.5    
    errh1 = compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,integration_weights,integration_points)

    print("L2 error: ",errl2)
    print("H1 error: ",errh1) 
    err_list_l2.append(errl2.item())
    err_list_h1.append(errh1.item())
    # plot_2D(my_model.cpu())  

d = 3 
print("L2 order:",0.5 + (2*relu_k +1)/(2*d))
print("H1 order:",0.5 + (2*(relu_k-1) +1)/(2*d))

output_convergence_order(actual_neuron_list,err_list_l2, err_list_h1)

