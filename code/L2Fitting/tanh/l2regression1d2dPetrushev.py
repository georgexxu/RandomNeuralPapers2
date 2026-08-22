"""1D/2D discrete ell^2 regression (collocation) with tanh Petrushev scheme.

Mirrors the structure of l2regression1d2dRFM.py, but initializes (w, b) via
initialize_w_b_petrushev (see l2regression-nd-tanh-petrushev.ipynb):
  - run experiments, save npz (with simple seeds)
  - replot 1D and 2D figures separately from saved data
"""
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext
from pathlib import Path
import sys

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

pi = torch.tensor(np.pi, dtype=torch.float64)
torch.set_default_dtype(torch.float64)

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_CODE_ROOT))
from utils_quad_init import (
    PiecewiseGQ1D_weights_points,
    PiecewiseGQ2D_weights_points,
    initialize_w_b_petrushev,
    model_tanh,
)

DATA_DIR = Path(__file__).resolve().parent / 'data'
M_LIST = [1, 2, 4]
R_M_LIST = [1, 2, 4, 8]
# Notebook defaults: 1D radius=4 (cell 9), 2D radius=2.5 (cell 18)
RADIUS_1D = 4.0
RADIUS_2D = 2.5
BASE_SEED = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assemble_least_square_linear_system_quadrature(
    model, rhs, integration_weights, integration_points,
):
    """Assemble design matrix D and rhs for discrete least squares."""
    W = model.fc1.weight.data
    b = model.fc1.bias.data
    Z = integration_points @ W.t() + b
    D = torch.tanh(Z) * integration_weights
    b_rhs = rhs(integration_points) * integration_weights
    return D, b_rhs


def rescale_mat_rhs(A, f):
    c = 100.0
    for i in range(len(A)):
        max_a = abs(A[i, :]).max()
        max_b = A[i, :].max()
        if max_a != max_b:
            ratio = -c / max_a
        else:
            ratio = c / max_a
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio
    return A, f


def npz_path_1d(m):
    return DATA_DIR / f'petrushev_1d_sin_m_{m}_collocation.npz'


def npz_path_2d(m):
    return DATA_DIR / f'petrushev_2d_sin_m_{m}_collocation.npz'


def run_exp_1d(
    m, R_m_list=None, trial_num=5, base_seed=BASE_SEED, radius=RADIUS_1D,
):
    """1D collocation / discrete ell^2 regression (Petrushev); saves npz."""
    R_m_list = list(R_m_list or R_M_LIST)
    Nx, order, Q = 1024, 5, 1024
    integration_weights, integration_points = PiecewiseGQ1D_weights_points(
        -1, 1, Nx=Nx, order=order,
    )
    collocation_points = torch.linspace(-1, 1, Q).view(-1, 1)
    collocation_weights = torch.ones(Q).view(-1, 1)
    neuron_num_arr = np.array([2 ** i for i in range(2, 11)], dtype=float)

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
                my_model = initialize_w_b_petrushev(
                    my_model, R_m=R_m, radius=radius,
                )
                D, b_rhs = assemble_least_square_linear_system_quadrature(
                    my_model, target, collocation_weights, collocation_points,
                )
                D, b_rhs = rescale_mat_rhs(D, b_rhs)
                sol = scipy.linalg.lstsq(
                    D.cpu().numpy(), b_rhs.cpu().numpy(),
                )[0]
                sol = torch.from_numpy(sol).to(device)
                my_model.fc2.weight.data[0, :] = sol.view(-1)
                with torch.no_grad():
                    nn_func_values = my_model(integration_points)
                target_values = target(integration_points)
                err_l2 = (
                    integration_weights.t() @ (nn_func_values - target_values) ** 2
                ) ** 0.5
                err_l2_trial[trial] = err_l2
            mean_err_l2_list.append(float(torch.mean(err_l2_trial)))
            seed_n_list.append(seeds_trial)
            print(
                f'1d-petrushev m={m} r1={radius} r2={R_m} n={n}: '
                f'L2={mean_err_l2_list[-1]:.3e}',
                flush=True,
            )
        mean_err_rows.append(mean_err_l2_list)
        seed_rows.append(seed_n_list)

    mean_err_l2 = np.asarray(mean_err_rows, dtype=float)
    seed_arr = np.asarray(seed_rows, dtype=np.int64)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = npz_path_1d(m)
    np.savez(
        out,
        d=1,
        m=m,
        Nx=Nx,
        order=order,
        Q=Q,
        trial_num=trial_num,
        base_seed=base_seed,
        radius=radius,
        R_m_arr=np.asarray(R_m_list, dtype=float),
        neuron_num_arr=neuron_num_arr,
        mean_err_l2=mean_err_l2,
        seed_arr=seed_arr,
    )
    print(f'saved {out}', flush=True)
    return out


def run_exp_2d(
    m, R_m_list=None, trial_num=5, base_seed=BASE_SEED, radius=RADIUS_2D,
):
    """2D collocation / discrete ell^2 regression (Petrushev); saves npz."""
    R_m_list = list(R_m_list or R_M_LIST)
    Nx, order = 50, 3
    integration_weights_test, integration_points_test = PiecewiseGQ2D_weights_points(
        Nx, order, [-1, -1], [1, 1],
    )
    x = torch.linspace(-1, 1, 50)
    X1, X2 = torch.meshgrid(x, x, indexing='ij')
    x_train = torch.concat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
    collocation_weights = 1.0 / x_train.size(0)
    collocation_points = x_train
    neuron_num_arr = np.array([2 ** i for i in range(2, 12)], dtype=float)

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
                my_model = initialize_w_b_petrushev(
                    my_model, R_m=R_m, scale=1, radius=radius,
                )
                D, b_rhs = assemble_least_square_linear_system_quadrature(
                    my_model, target, collocation_weights, collocation_points,
                )
                D, b_rhs = rescale_mat_rhs(D, b_rhs)
                sol = scipy.linalg.lstsq(
                    D.cpu().numpy(), b_rhs.cpu().numpy(),
                )[0]
                sol = torch.from_numpy(sol).to(device)
                my_model.fc2.weight.data[0, :] = sol.view(-1)
                with torch.no_grad():
                    nn_func_values = my_model(integration_points_test)
                target_values = target(integration_points_test)
                err_l2 = (
                    integration_weights_test.t()
                    @ (nn_func_values - target_values) ** 2
                ) ** 0.5
                err_l2_trial[trial] = err_l2
            mean_err_l2_list.append(float(torch.mean(err_l2_trial)))
            seed_n_list.append(seeds_trial)
            print(
                f'2d-petrushev m={m} r1={radius} r2={R_m} n={n}: '
                f'L2={mean_err_l2_list[-1]:.3e}',
                flush=True,
            )
        mean_err_rows.append(mean_err_l2_list)
        seed_rows.append(seed_n_list)

    mean_err_l2 = np.asarray(mean_err_rows, dtype=float)
    seed_arr = np.asarray(seed_rows, dtype=np.int64)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = npz_path_2d(m)
    np.savez(
        out,
        d=2,
        m=m,
        Nx=Nx,
        order=order,
        Q_grid=50,
        trial_num=trial_num,
        base_seed=base_seed,
        radius=radius,
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
    radius = float(data['radius']) if 'radius' in data.files else None
    for i, R_m in enumerate(R_m_arr):
        if radius is not None:
            label = rf'$r_1={radius:g},\ r_2={int(R_m)}$'
        else:
            label = rf'$R={int(R_m)}$'
        ax.plot(x, mean_err_l2[i], '.-', label=label)
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


def plot_1d(m_list=None, out_name='Petrushev_1d_l2reg.png'):
    m_list = list(m_list or M_LIST)
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=400)
    for ax, m in zip(axes, m_list):
        title = rf"Petrushev. $u=\sin(m\pi x)$, $m={m}$"
        plot_panel(ax, npz_path_1d(m), title)
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / out_name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}', flush=True)
    return out


def plot_2d(m_list=None, out_name='Petrushev_2d_l2reg.png'):
    m_list = list(m_list or M_LIST)
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=400)
    for ax, m in zip(axes, m_list):
        title = rf"Petrushev. $u=\sin(m\pi x_1)\sin(m\pi x_2)$, $m={m}$"
        plot_panel(ax, npz_path_2d(m), title)
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / out_name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}', flush=True)
    return out


def plot_all(m_list=None):
    return plot_1d(m_list), plot_2d(m_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='1D/2D tanh discrete ell^2 regression (collocation) Petrushev',
    )
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
    plot_all()
