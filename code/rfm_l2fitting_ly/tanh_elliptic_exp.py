import torch 
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np 
import itertools
import scipy 
import math
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
pi = torch.tensor(np.pi,dtype=torch.float64)

from utils import SquareMesh2D_points
from utils import uniform_rand_init
from utils import plot_tanh_err_convergence_levels
from model import ShallowTanhSolver
from model import PouShallowTanhSolver


def run_exp(s, m, l, nmin=6, nmax=13, neuron_max=2500):
    x_train = SquareMesh2D_points(0,1,100)
    x_test = SquareMesh2D_points(0,1,150)

    if l == 0:
        # Global fitter
        neuron_num_arr= np.array([2**i for i in range(nmin, nmax)]) 
        neuron_arr = []
        rel_errl2_arr = []
        for i, n in enumerate(neuron_num_arr):
            ws, bs = uniform_rand_init(n, s=s)
            fitter = ShallowTanhSolver(
                ws, bs, x_train, m1=m, m2=m)
            if fitter.n > neuron_max:
                break
            fitter.solve()
            rel_errl2 = fitter.eval(x_test)
            neuron_arr.append(fitter.n)
            rel_errl2_arr.append(rel_errl2)
            print('num neuron : {:} - rel l2 : {:.4e}'.format(fitter.n, rel_errl2))
    else:
        neuron_num_arr= np.array([2**i for i in range(2,11)]) 
        neuron_arr = []
        rel_errl2_arr = []
        for i, n in enumerate(neuron_num_arr):
            ws, bs = uniform_rand_init(n, s=s)
            fitter = PouShallowTanhSolver(
                ws, bs, x_train, l=l, m1=m, m2=m)
            if fitter.J * fitter.n > neuron_max:
                break 
            fitter.solve()
            rel_errl2 = fitter.eval(x_test)
            neuron_arr.append(fitter.n*fitter.J)
            rel_errl2_arr.append(rel_errl2)
            print('local {:} num neuron : {:} - rel l2 : {:.4e}'.format(fitter.J, fitter.n*fitter.J, rel_errl2))

    return np.array(neuron_arr), np.array(rel_errl2_arr)

if __name__ == '__main__':

    s_list = [1, 2, 4, 8]
    m_list = [1, 2, 4]
    l_list = [0, 1, 2, 3]
    neurons_dict = {}
    errs_dict = {}
    for s in s_list:
        for m in m_list:
            print(f'start exp (l2fit) : s {s} - m {m}')
            neuron_dict_outpath = f'./logs/elliptic_neurons_s{s}_m{m}.npy'
            err_dict_outpath = f'./logs/elliptic_errs_s{s}_m{m}.npy'

            if os.path.exists(neuron_dict_outpath) & os.path.exists(err_dict_outpath):
                neurons_dict = np.load(neuron_dict_outpath, allow_pickle=True).item()
                errs_dict = np.load(err_dict_outpath, allow_pickle=True).item()
            else:
                for l in l_list:
                    neurons, errs = run_exp(s, m, l, 2, 13)
                    neurons_dict[l] = neurons
                    errs_dict[l] = errs

                np.save(neuron_dict_outpath, neurons_dict)
                np.save(err_dict_outpath, errs_dict)
            title = "$-\Delta u + u = f$. Tanh-${}$. $u(x_1, x_2) = \sin(m_1\pi x_1) \sin(m_2\pi x_2)$, $m_1,m_2 = {}, {}$".format(s, m, m)
            outpath = f'./figures/elliptic_tanh{s}_m{m}.jpg'
            print(f'save figure {outpath}')
            plot_tanh_err_convergence_levels(scale=s, d=2,
                            neuron_dict=neurons_dict,
                            err_dict=errs_dict,
                            levels=l_list,
                            title=title,
                            outpath=outpath)
