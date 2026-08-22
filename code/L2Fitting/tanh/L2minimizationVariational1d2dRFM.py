## This version of 1D greedy algorithm, the numerical quadrature used for assembling the linear system is consistent with 
## the quadrature used for the argmax subproblem.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext
import time
import math 
import scipy 
from scipy.sparse import linalg
from pathlib import Path
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 

pi = torch.tensor(np.pi,dtype=torch.float64)
torch.set_default_dtype(torch.float64)

import sys, os
_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_CODE_ROOT))
from utils_quad_init import PiecewiseGQ1D_weights_points, PiecewiseGQ2D_weights_points, PiecewiseGQ3D_weights_points, initialize_w_b_uniform, initialize_w_b_sphere, initialize_w_b_petrushev
from utils_quad_init import model_tanh 

def minimize_linear_layer_explicit_assemble(model,target,weights, integration_points,solver="direct",activation = 'tanh'):
    """
    1. assemble matrix and rhs vector 
    2. solve the linear system and return the solution 
    """
    # Nx = 1024
    # order = 5
    # weights, integration_points = PiecewiseGQ1D_weights_points(-1,1,Nx,order) 
    # start_time = time.time()
    w = model.fc1.weight.data
    if activation == 'cos':
        basis_value_col = torch.cos(2*pi* (integration_points @ w.t()) ) # activation function dependent 
    elif activation == 'tanh':
        b = model.fc1.bias.data 
        basis_value_col = torch.tanh(integration_points @ w.t() + b) # activation function dependent 
    weighted_basis_value_col = basis_value_col * weights
    jac = weighted_basis_value_col.t() @ basis_value_col 
    # print("assembling the matrix time taken: ", time.time()-start_time) 
    rhs = weighted_basis_value_col.t() @ target(integration_points) 

    # print("using: ",solver, end = " ")
    if solver == "cg": 
        sol, exit_code = linalg.cg(np.array(jac.detach().cpu()),np.array(rhs.detach().cpu()),tol=1e-12)
        sol = torch.tensor(sol).view(1,-1)
    elif solver == "direct": 
#         sol = np.linalg.inv( np.array(jac.detach().cpu()) )@np.array(rhs.detach().cpu())
        sol = (torch.linalg.solve( jac.detach(), rhs.detach())).view(1,-1)
            # A x = b with A SPD

        ## regularizer 
        # lam = 1e-10
        # L = torch.linalg.cholesky(jac.detach() +  lam*torch.eye(jac.size(-1)) )                  # A = L L^T
        # sol = torch.cholesky_solve(rhs.detach(), L).view(1, -1)  # exact solve
        # Q, R = torch.linalg.qr(jac.detach(), mode="reduced")     # A = Q R
        # y = Q.mH @ rhs
        # sol = torch.linalg.solve_triangular(R, y,upper=True).view(1, -1)
    elif solver == "ls":
        sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd',rcond = 1e-13).solution).view(1,-1)
        # sol = (torch.linalg.lstsq(jac.detach(),rhs.detach()).solution).view(1,-1) # gpu/cpu, driver = 'gels', cannot solve singular
    return sol 

def minimize_linear_layer_explicit_assemble_mat(model,target,weights, integration_points,solver="direct",activation = 'tanh'):
    """
    1. assemble matrix and rhs vector 
    2. solve the linear system and return the solution 
    """
    # Nx = 1024
    # order = 5
    # weights, integration_points = PiecewiseGQ1D_weights_points(-1,1,Nx,order) 
    # start_time = time.time()
    w = model.fc1.weight.data
    if activation == 'cos':
        basis_value_col = torch.cos(2*pi* (integration_points @ w.t()) ) # activation function dependent 
    elif activation == 'tanh':
        b = model.fc1.bias.data 
        basis_value_col = torch.tanh(integration_points @ w.t() + b) # activation function dependent 
    weighted_basis_value_col = basis_value_col * weights
    jac = weighted_basis_value_col.t() @ basis_value_col 
    # print("assembling the matrix time taken: ", time.time()-start_time) 
    rhs = weighted_basis_value_col.t() @ target(integration_points) 

    return jac, rhs 

def rescale_mat_rhs(A,f):
    c = 100.0
    for i in range(len(A)):
        ## change 
        max_a = abs(A[i,:]).max()
        max_b = A[i,:].max()
        if max_a != max_b: 
            ratio = -c/max_a
            A[i,:] = A[i,:]*ratio
            f[i] = f[i]*ratio
        else: 
            ratio = c/max_a
            A[i,:] = A[i,:]*ratio
            f[i] = f[i]*ratio
    return A,f 
def minimize_linear_layer_least_square(model,target,weights, integration_points,solver="direct",activation = 'tanh'):
    """
    1. assemble matrix and rhs vector 
    2. solve the linear system and return the solution 
    """
    # Nx = 1024
    # order = 5
    # weights, integration_points = PiecewiseGQ1D_weights_points(-1,1,Nx,order) 
    # start_time = time.time()
    w = model.fc1.weight.data
    if activation == 'cos':
        basis_value_col = torch.cos(2*pi* (integration_points @ w.t()) ) # activation function dependent 
    elif activation == 'tanh':
        b = model.fc1.bias.data 
        basis_value_col = torch.tanh(integration_points @ w.t() + b) # activation function dependent 
    weighted_basis_value_col = basis_value_col * (weights**0.5)
    jac = weighted_basis_value_col 
    # print("assembling the matrix time taken: ", time.time()-start_time) 
    rhs =  target(integration_points) * (weights**0.5)
    # jac, rhs = rescale_mat_rhs(jac,rhs) 
    # sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd').solution).view(1,-1) 
    sol = scipy.linalg.lstsq(jac.cpu().numpy(),rhs.cpu().numpy())[0]
    sol = torch.from_numpy(sol).to(device).view(1,-1)
    return sol 

def linear_layer_least_square_mat(model,target,weights, integration_points,activation = 'tanh'):
    """
    1. assemble matrix and rhs vector 
    2. solve the linear system and return the solution 
    """
    w = model.fc1.weight.data
    if activation == 'cos':
        basis_value_col = torch.cos(2*pi* (integration_points @ w.t()) ) # activation function dependent 
    elif activation == 'tanh':
        b = model.fc1.bias.data 
        basis_value_col = torch.tanh(integration_points @ w.t() + b) # activation function dependent 
    weighted_basis_value_col = basis_value_col * (weights**0.5)
    jac = weighted_basis_value_col 
    # print("assembling the matrix time taken: ", time.time()-start_time) 
    rhs =  target(integration_points) * (weights**0.5)
    # jac, rhs = rescale_mat_rhs(jac,rhs) 
    # sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd').solution).view(1,-1) 

    return jac 

DATA_DIR = Path(__file__).resolve().parent / 'data'
M_LIST = [1, 2, 4]
R_M_LIST = [1, 2, 4, 8]
BASE_SEED = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def npz_path_1d(m):
    return DATA_DIR / f'ELM_1d_sin_m_{m}_variational.npz'


def npz_path_2d(m):
    return DATA_DIR / f'elm_2d_sin_m_{m}_variational.npz'


def run_exp_1d(m, R_m_list=None, trial_num=5, base_seed=BASE_SEED):
    """1D L2 minimization; returns and saves arrays (no plotting)."""
    R_m_list = list(R_m_list or R_M_LIST)
    Nx, order = 1024, 5
    weights, integration_points = PiecewiseGQ1D_weights_points(-1, 1, Nx=Nx, order=order)
    neuron_num_arr = np.array([2 ** i for i in range(2, 9)], dtype=float)

    def target(x):
        return torch.sin(m * pi * x)

    mean_err_rows = []
    seed_rows = []
    seed_counter = base_seed
    for R_m in R_m_list:
        mean_err_l2_list = []
        seed_n_list = []
        for n in neuron_num_arr.astype(int):
            err_l2_trial = torch.zeros(trial_num)
            seeds_trial = np.zeros(trial_num, dtype=np.int64)
            for trial in range(trial_num):
                seed = seed_counter
                seed_counter += 1
                seeds_trial[trial] = seed
                set_seed(seed)
                my_model = model_tanh(1, n, 1).to(device)
                my_model = initialize_w_b_uniform(my_model, R_m)
                sol = minimize_linear_layer_explicit_assemble(
                    my_model, target, weights, integration_points,
                    solver='direct', activation='tanh',
                )
                my_model.fc2.weight.data[0, :] = sol[:]
                with torch.no_grad():
                    nn_func_values = my_model(integration_points)
                target_values = target(integration_points)
                err_l2 = (weights.t() @ (nn_func_values - target_values) ** 2) ** 0.5
                err_l2_trial[trial] = err_l2
            mean_err_l2_list.append(float(torch.mean(err_l2_trial)))
            seed_n_list.append(seeds_trial)
            print(f'1d m={m} R={R_m} n={n}: L2={mean_err_l2_list[-1]:.3e}', flush=True)
        mean_err_rows.append(mean_err_l2_list)
        seed_rows.append(seed_n_list)

    mean_err_l2 = np.asarray(mean_err_rows, dtype=float)
    seed_arr = np.asarray(seed_rows, dtype=np.int64)  # (n_R, n_neurons, trial_num)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = npz_path_1d(m)
    np.savez(
        out,
        d=1,
        m=m,
        Nx=Nx,
        order=order,
        trial_num=trial_num,
        base_seed=base_seed,
        R_m_arr=np.asarray(R_m_list, dtype=float),
        neuron_num_arr=neuron_num_arr,
        mean_err_l2=mean_err_l2,
        seed_arr=seed_arr,
    )
    print(f'saved {out}', flush=True)
    return out


def run_exp_2d(m, R_m_list=None, trial_num=5, exponent=10, base_seed=BASE_SEED):
    """2D L2 minimization with target sin(m π x1) sin(m π x2); saves npz."""
    R_m_list = list(R_m_list or R_M_LIST)
    Nx, order = 100, 3
    weights, integration_points = PiecewiseGQ2D_weights_points(
        Nx=Nx, order=order, l_point=(-1, -1), r_point=(1, 1),
    )
    neuron_num_arr = np.array([2 ** i for i in range(3, exponent + 1)], dtype=float)

    def target(x):
        return torch.sin(m * pi * x[:, 0:1]) * torch.sin(m * pi * x[:, 1:2])

    mean_err_rows = []
    seed_rows = []
    seed_counter = base_seed
    for R_m in R_m_list:
        mean_err_l2_list = []
        seed_n_list = []
        for n in neuron_num_arr.astype(int):
            err_l2_trial = torch.zeros(trial_num)
            seeds_trial = np.zeros(trial_num, dtype=np.int64)
            for trial in range(trial_num):
                seed = seed_counter
                seed_counter += 1
                seeds_trial[trial] = seed
                set_seed(seed)
                my_model = model_tanh(2, n, 1).to(device)
                my_model = initialize_w_b_uniform(my_model, R_m=R_m)
                sol = minimize_linear_layer_explicit_assemble(
                    my_model, target, weights, integration_points, solver='direct',
                )
                my_model.fc2.weight.data[0, :] = sol[:]
                with torch.no_grad():
                    nn_func_values = my_model(integration_points)
                target_values = target(integration_points)
                err_l2 = (weights.t() @ (nn_func_values - target_values) ** 2) ** 0.5
                err_l2_trial[trial] = err_l2
            mean_err_l2_list.append(float(torch.mean(err_l2_trial)))
            seed_n_list.append(seeds_trial)
            print(f'2d m={m} R={R_m} n={n}: L2={mean_err_l2_list[-1]:.3e}', flush=True)
        mean_err_rows.append(mean_err_l2_list)
        seed_rows.append(seed_n_list)

    mean_err_l2 = np.asarray(mean_err_rows, dtype=float)
    seed_arr = np.asarray(seed_rows, dtype=np.int64)  # (n_R, n_neurons, trial_num)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = npz_path_2d(m)
    np.savez(
        out,
        d=2,
        m=m,
        Nx=Nx,
        order=order,
        trial_num=trial_num,
        base_seed=base_seed,
        R_m_arr=np.asarray(R_m_list, dtype=float),
        neuron_num_arr=neuron_num_arr,
        mean_err_l2=mean_err_l2,
        seed_arr=seed_arr,
    )
    print(f'saved {out}', flush=True)
    return out


def _style_ax(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'Number of neurons $n$')
    ax.set_ylabel(r'$L^2$ error')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(True, which='both', ls=':', alpha=0.6)


def plot_panel(ax, path, title):
    data = np.load(path)
    x = np.asarray(data['neuron_num_arr'], dtype=float)
    R_m_arr = np.asarray(data['R_m_arr'], dtype=float)
    mean_err_l2 = np.asarray(data['mean_err_l2'], dtype=float)
    for i, R_m in enumerate(R_m_arr):
        ax.plot(x, mean_err_l2[i], '.-', label=rf'$R={int(R_m)}$')
    ax.set_xlim(x.min() / 1.15, x.max() * 1.15)
    ax.set_title(title)
    _style_ax(ax)


def _apply_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 11,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
        'axes.linewidth': 1.1,
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
    })


def plot_1d(m_list=None, out_name='ELM_1d_L2min.png'):
    """1x3 figure for 1D m=1,2,4."""
    m_list = list(m_list or M_LIST)
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=400)
    for ax, m in zip(axes, m_list):
        title = rf'$u=\sin(m\pi x)$, $m={m}$'
        plot_panel(ax, npz_path_1d(m), title)
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / out_name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}', flush=True)
    return out


def plot_2d(m_list=None, out_name='ELM_2d_L2min.png'):
    """1x3 figure for 2D m=1,2,4."""
    m_list = list(m_list or M_LIST)
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=400)
    for ax, m in zip(axes, m_list):
        title = rf'$u=\sin(m\pi x_1)\sin(m\pi x_2)$, $m={m}$'
        plot_panel(ax, npz_path_2d(m), title)
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / out_name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}', flush=True)
    return out


def plot_combined_1d2d(m_list=None):
    """Save separate 1D and 2D figures."""
    return plot_1d(m_list), plot_2d(m_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='1D/2D tanh L2 minimization RFM')
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Skip experiments; replot from saved npz files',
    )
    args = parser.parse_args()

    if not args.plot_only:
        for m in M_LIST:
            run_exp_1d(m)
        for m in M_LIST:
            run_exp_2d(m)
    plot_combined_1d2d()
