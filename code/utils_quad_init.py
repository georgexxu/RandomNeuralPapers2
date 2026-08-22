import torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math 
import time
import sys
import scipy 
from scipy.sparse import linalg
from pathlib import Path
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 
pi = torch.tensor(np.pi,dtype=torch.float64)
torch.set_default_dtype(torch.float64)

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

class model_tanh(nn.Module):
    """ ReLU k shallow neural network
    Parameters: 
    input size: input dimension
    hidden_size1 : number of hidden layers 
    num_classes: output classes 
    k: degree of relu functions
    """
    def __init__(self, input_size, hidden_size1, num_classes):
        super().__init__()
        self.activation = 'tanh'
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, num_classes,bias = False)
    def forward(self, x):
        u1 = self.fc2(torch.tanh(self.fc1(x)))
        return u1
    def hidden(self,x):
        return torch.tanh(self.fc1(x)) 
    def evaluate_derivative(self, x, i):
        """
        Evaluate the derivative of the network output with respect to the i-th input.
        Note: i is 1-indexed as in the original code.
        """

        fprime = 1 - torch.tanh(x @ self.fc1.weight.t() + self.fc1.bias) ** 2
        # Adjust index: i-1 because original code is 1-indexed.
        u1 = (fprime * self.fc1.weight.t()[i-1:i, :]) @ self.fc2.weight.t()
        return u1

class model_cosine(nn.Module):
    """ ReLU k shallow neural network
    Parameters: 
    input size: input dimension
    hidden_size1 : number of hidden layers 
    num_classes: output classes 
    k: degree of relu functions
    """
    def __init__(self, input_size, hidden_size1, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1, bias = False)
        self.fc2 = nn.Linear(hidden_size1, num_classes,bias = False)
    def forward(self, x):
        u1 = self.fc2(torch.cos(2 * pi * self.fc1(x)))
        return u1

def PiecewiseGQ1D_weights_points(x_l,x_r,Nx, order):
    """ Output the coeffients and weights for piecewise Gauss Quadrature 
    Parameters
    ----------
    x_l : float 
    left endpoint of an interval 
    
    x_r: float
    right endpoint of an interval 
    
    integration_intervals: int
    number of subintervals for integration
    
    Returns
    -------
    coef1_expand
    
    gw_expand
    
    integration_points
    """
    x,w = np.polynomial.legendre.leggauss(order)
    gx = torch.tensor(x).to(device)
    gx = gx.view(1,-1) # row vector 
    gw = torch.tensor(w).to(device)    
    gw = gw.view(-1,1) # Column vector 
    nodes = torch.linspace(x_l,x_r,Nx+1).view(-1,1).to(device) 
    coef1 = ((nodes[1:,:] - nodes[:-1,:])/2) # n by 1  
    coef2 = ((nodes[1:,:] + nodes[:-1,:])/2) # n by 1  
    coef2_expand = coef2.expand(-1,gx.size(1)) # Expand to n by p shape, -1: keep the first dimension n , expand the 2nd dim (columns)
    integration_points = coef1@gx + coef2_expand
    integration_points = integration_points.flatten().view(-1,1) # Make it a column vector
    gw_expand = torch.tile(gw,(Nx,1)) # rows: n copies of current tensor, columns: 1 copy, no change
    # Modify coef1 to be compatible with func_values
    coef1_expand = coef1.expand(coef1.size(0),gx.size(1))    
    coef1_expand = coef1_expand.flatten().view(-1,1)

    return coef1_expand.to(device)*gw_expand.to(device), integration_points.to(device)


def PiecewiseGQ2D_weights_points(Nx, order,l_point = (-1,-1), r_point = (1,1)): 
    """ A slight modification of PiecewiseGQ2D function that only needs the weights and integration points.
    Parameters. 
    Only works for the square domain centered at origin or with the left point at origin.
    ----------
    Nx: int 
        number of intervals along the dimension. No Ny, assume Nx = Ny
    order: int 
        order of the Gauss Quadrature
    l_point: (x1,x2)
        left bottom point of the rectangular domain 
    r_point: (x1,x2)
        right top point of the rectangular domain 

    Returns
    -------
    long_weights: torch.tensor
    integration_points: torch.tensor
    """
    assert Nx%2==0, "Nx should be even" 

    h1 = (r_point[0] - l_point[0])/Nx
    h2 = (r_point[1] - l_point[1])/Nx 

    x, w = np.polynomial.legendre.leggauss(order)
    gauss_pts = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T
    weights =  (w*w[:,None]).ravel()

    gauss_pts =torch.tensor(gauss_pts)
    weights = torch.tensor(weights)

    long_weights =  torch.tile(weights,(Nx**2,1))
    long_weights = long_weights.reshape(-1,1)
    long_weights = long_weights * (h1/2) * (h2/2) # scale the weights  
    integration_points = torch.tile(gauss_pts,(Nx**2,1))
    scale_factor = h1/2 
    integration_points = scale_factor * integration_points

    offset = 0 if l_point[0] == 0 else 1 

    index = np.arange(1,Nx+1)- (Nx/2)*offset - 0.5 
    ordered_pairs = np.array(np.meshgrid(index,index,indexing='ij'))
    ordered_pairs = ordered_pairs.reshape(2,-1).T
    ordered_pairs = torch.tensor(ordered_pairs)
    ordered_pairs = torch.tile(ordered_pairs, (1,order**2)) # number of GQ points
    ordered_pairs =  ordered_pairs.reshape(-1,2)

    translation = ordered_pairs*h1
    integration_points = integration_points + translation 
    return long_weights, integration_points

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


def Monte_Carlo_dDim_weights_points(M, d,bl = -1,ur = 1):
    # Samples N points uniformly in Ω at t = 0 (initial condition)
    length = ur - bl
    vol = length ** d
    integration_points = torch.empty(M, d).uniform_(bl, ur)
    # t = torch.zeros(N, 1)
    # return torch.cat([X, t], dim=1).to(device)
    weights = torch.ones(M,1).to(device)/M * vol 
    return weights.to(device), integration_points.to(device)  

def MonteCarlo_Sobol_dDim_weights_points(M ,d = 4,bl = -1,ur = 1):
    
    length = ur - bl
    vol = length ** d 
    Sob_integral = torch.quasirandom.SobolEngine(dimension =d, scramble= False, seed=None) 
    integration_points = Sob_integral.draw(M).double() 
    integration_points = integration_points.to(device) * (length) - length/2
    weights = torch.ones(M,1).to(device)/M * vol 
    return weights.to(device), integration_points.to(device) 

def plot_solution_modified(r1,r2,model,x_test,u_true,name=None): 
    # Plot function: test results 
    u_model_cpu = model(x_test).cpu().detach()
    
    w = model.fc1.weight.data.squeeze()
    b = model.fc1.bias.data.squeeze()
    x_model_pt = (-b/w).view(-1,1)
    u_model_pt = model(x_model_pt).cpu().detach()
    plt.figure(dpi = 100)
    plt.plot(x_test.cpu(),u_model_cpu,'-.',label = "nn function")
    plt.plot(x_test.cpu(),u_true.cpu(),label = "true")
#     plt.plot(x_model_pt,u_model_pt,'.r')
    if name!=None: 
        plt.title(name)
    plt.legend()
    plt.show()

def initialize_uniform(my_model,target = None):
    """
    mutator function. input: my_model
    """
    #FEM initialization
    neuron_number = my_model.fc1.bias.size(0)
    h = 2*1/neuron_number
    #wi(x-x_i)
    wi = torch.full((neuron_number,1),1/h)
    bi = torch.linspace(-1,1,neuron_number + 1)[:-1].view(neuron_number,1) #x_i
    bi = -(wi*bi).view(neuron_number)
    my_model.fc1.weight.data = wi # somehow this local variable is not destroyed
    my_model.fc1.bias.data = bi
    return my_model 
    
def initialize_w_b_uniform(my_model,R_m): # mostly used in ELM 
    nn.init.uniform_(my_model.fc1.weight, a = -R_m, b = R_m)
    nn.init.uniform_(my_model.fc1.bias, a = -R_m, b = R_m)
    return my_model 


def initialize_w_b_sphere(my_model,R_m):
    # generate a uniform grid on S^2 
    neuron_nums = my_model.fc1.bias.size(0) 
    out_feats, in_feats = my_model.fc1.weight.shape 
    if in_feats == 2: 
        indices = torch.arange(0, neuron_nums, dtype=torch.float) + 0.5
        phi = torch.acos(1 - 2*indices/neuron_nums)
        theta = pi * (1 + 5**0.5) * indices
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        points = torch.stack((x, y, z), dim=1)
        my_model.fc1.weight.data[:,:] = points[:,0:2] * R_m
        my_model.fc1.bias.data[:] = points[:,2] * R_m 
    elif in_feats == 1:
        # Match l2regression-nd-tanh-petrushev.ipynb: points on the semicircle
        theta = torch.linspace(0, pi, neuron_nums + 1)[:-1]
        w1 = torch.cos(theta) * R_m
        b = torch.sin(theta) * R_m
        my_model.fc1.weight.data[:, 0] = w1[:]
        my_model.fc1.bias.data[:] = b[:]
    else: 
        w_b = torch.randn(neuron_nums,in_feats + 1) 
        w_b = w_b / torch.norm(w_b,dim=1).view(-1,1) 
        w_b = w_b * R_m 
        my_model.fc1.weight.data[:,:] = w_b[:,:-1] 
        my_model.fc1.bias.data[:] = w_b[:,-1] 
    return my_model 

def initialize_w_b_petrushev(my_model,R_m,radius = 1, scale = 1):
    w = my_model.fc1.weight
    b = my_model.fc1.bias
    out_feats, in_feats = w.shape
    device = w.device
    n1 = scale * math.ceil(out_feats**((in_feats - 1)/in_feats)) 
    n2 = math.ceil(out_feats**(1/in_feats))
    with torch.no_grad():
        if in_feats == 1:  #n by 1 
            weights = torch.ones(n1,1).to(device)
            # weights[:,0] = torch.linspace(-R_m,R_m,n1) 
        elif in_feats == 2:
            weights = torch.zeros(n1,2).to(device)
            theta = np.linspace(0,pi,n1,endpoint=False) 
            theta = torch.tensor(theta).to(device)
            weights[:, 0] = theta.cos()
            weights[:, 1] = theta.sin() 
            #form a tensor product of weights and biases
        elif in_feats == 3: 
            indices = torch.arange(0, n1, dtype=torch.float) + 0.5
            phi = torch.acos(1 - 2*indices/n1)
            theta = pi * (1 + 5**0.5) * indices
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)
            weights = torch.stack((x, y, z), dim=1)
        elif in_feats >= 4: 
            # nn.init.normal_(w, mean = 0, std = 1)
            weights = torch.randn(n1, in_feats).to(device) 
            weights = weights / torch.norm(weights,dim=1).view(-1,1) 

        # form a tensor product of weights and biases 
        weights = torch.tile(weights, (1,n2)).reshape(-1,in_feats) * radius 
        # print("weights shape: ", weights.shape) 
        # print("nn w shape: ", w.shape)
        my_model = model_tanh(in_feats, n1*n2, 1).to(device) 
        biases = torch.linspace(-R_m + 0.01,R_m,n2).to(device)
        # biases = torch.tensor(biases).to(device) 
        biases = torch.tile(biases, (n1,1)).reshape(-1) 
        # w[:,:] = weights[:out_feats,:] 
        my_model.fc1.weight.data[:,:] = weights[:,:]
        my_model.fc1.bias.data[:] = biases[:] 

    return my_model 

def initialize_normal(my_model, mu, sigma):
    nn.init.normal_(my_model.fc1.weight, mean = mu, std = sigma)
    return my_model

def set_linear_layer_to_fixed_weights(my_model):
    neuron_num = my_model.fc1.weight.size(0)
    coefs = torch.ones(neuron_num) * 1/neuron_num 
    my_model.fc2.weight.data[0,:] = coefs[:]  
    return my_model