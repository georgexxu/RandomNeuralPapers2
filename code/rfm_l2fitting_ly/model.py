import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 

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

    def initialize_model(self):
        # generate a uniform grid on S^2 
        neuron_nums = self.fc1.bias.size(0) 

        indices = torch.arange(0, neuron_nums, dtype=torch.float) + 0.5
        phi = torch.acos(1 - 2*indices/neuron_nums)
        theta = pi * (1 + 5**0.5) * indices
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        points = torch.stack((x, y, z), dim=1)
        my_model.fc1.weight.data[:,:] = points[:,0:2]
        my_model.fc1.bias.data[:] = points[:,2]
        
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
    