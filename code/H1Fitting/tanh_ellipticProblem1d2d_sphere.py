"""1D/2D elliptic PDE collocation with tanh sphere scheme.

PDE: -Δu + u = f on Ω = (-1,1)^d, Dirichlet BC via collocation least squares.

Mirrors the driver style of l2regression1d2dSphere.py (run → npz → replot),
using the sphere-scheme snippets from ellipticProblem_collocation_sphere.ipynb:
  - 1D: main(m1,m2,m3) cell (Rm in {1,2,4,8})
  - 2D: tanh sphere experiment (Rm in {2,4,8,16})
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext
import scipy
from pathlib import Path
import sys

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

pi = torch.tensor(np.pi, dtype=torch.float64)
torch.set_default_dtype(torch.float64)

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_CODE_ROOT))
from utils_quad_init import (
    PiecewiseGQ1D_weights_points,
    PiecewiseGQ2D_weights_points,
)

DATA_DIR = Path(__file__).resolve().parent / 'data'
# Exact solution frequencies (notebook defaults)
M1, M2, M3 = 1, 2, 4
R_M_LIST_1D = [1, 2, 4, 8]
R_M_LIST_2D = [2, 4, 8, 16]
BASE_SEED = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ELMModel(nn.Module):
    """Shallow tanh ELM used by the notebook elliptic collocation solver."""

    def __init__(self, input_dim, M, Rm=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.M = M
        self.activation = 'tanh'
        self.k = 1  # unused for tanh; kept for design-matrix API
        self.W = nn.Parameter(
            torch.empty(M, input_dim).uniform_(-Rm, Rm), requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.empty(M).uniform_(-Rm, Rm), requires_grad=False,
        )
        self.phi = nn.Parameter(torch.zeros(1, M), requires_grad=False)

    def hidden(self, x):
        return torch.tanh(x @ self.W.t() + self.b)

    def forward(self, x):
        return self.hidden(x) @ self.phi.t()

    def evaluate_derivative(self, x, i):
        fprime = 1 - torch.tanh(x @ self.W.t() + self.b) ** 2
        return (fprime * self.W.t()[i - 1:i, :]) @ self.phi.t()

    def initialize_w_b_sphere(self, R_m):
        """Sphere / semicircle scheme (notebook cell)."""
        dev = self.W.device
        if self.input_dim == 2:
            indices = torch.arange(0, self.M, dtype=torch.float64, device=dev) + 0.5
            phi = torch.acos(1 - 2 * indices / self.M)
            theta = pi.to(dev) * (1 + 5 ** 0.5) * indices
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)
            points = torch.stack((x, y, z), dim=1)
            self.W = nn.Parameter(points[:, 0:2] * R_m, requires_grad=False)
            self.b = nn.Parameter(points[:, 2] * R_m, requires_grad=False)
        elif self.input_dim == 1:
            theta = torch.linspace(0, float(pi), self.M + 1, device=dev)[:-1]
            w1 = torch.cos(theta) * R_m
            b = torch.sin(theta) * R_m
            self.W = nn.Parameter(w1.view(-1, 1), requires_grad=False)
            self.b = nn.Parameter(b, requires_grad=False)
        else:
            w_b = torch.randn(self.M, self.input_dim + 1, device=dev)
            w_b = w_b / torch.norm(w_b, dim=1).view(-1, 1)
            w_b = w_b * R_m
            self.W = nn.Parameter(w_b[:, :-1], requires_grad=False)
            self.b = nn.Parameter(w_b[:, -1], requires_grad=False)
        return self


def generate_initial_points(N, d, x_l=-1, x_r=1, uniform_grid=True):
    """1D interior / collocation points (notebook)."""
    if uniform_grid:
        X = torch.linspace(x_l, x_r, N).view(-1, 1)
    else:
        X = torch.empty(N, d).uniform_(x_l, x_r)
    return X.to(device)


def generate_boundary_points(N_bc, d, x_l=-1, x_r=1):
    """Dirichlet boundary faces (notebook)."""
    points_list = []
    for i in range(d):
        for val in [x_l, x_r]:
            X = torch.empty(N_bc, d).uniform_(x_l, x_r)
            X[:, i] = val
            points_list.append(X)
    return torch.cat(points_list, dim=0).to(device)


def generate_initial_points_uniform_grid(N, d, x_l=-1, x_r=1):
    """2D tensor-product grid: interior + boundary (notebook)."""
    assert d == 2
    x = torch.linspace(x_l, x_r, N)
    x1, x2 = torch.meshgrid(x, x, indexing='ij')
    x1_int = x1[1:-1, 1:-1]
    x2_int = x2[1:-1, 1:-1]
    x_interior = torch.concat(
        [x1_int.reshape(-1, 1), x2_int.reshape(-1, 1)], dim=1,
    )
    x1_bc = torch.cat([
        x1[:1, :].reshape(-1, 1),
        x1[-1:, :].reshape(-1, 1),
        x1[1:N - 1, 0:1].reshape(-1, 1),
        x1[1:N - 1, -1:].reshape(-1, 1),
    ], dim=0)
    x2_bc = torch.cat([
        x2[:1, :].reshape(-1, 1),
        x2[-1:, :].reshape(-1, 1),
        x2[1:N - 1, 0:1].reshape(-1, 1),
        x2[1:N - 1, -1:].reshape(-1, 1),
    ], dim=0)
    x_boundary = torch.concat(
        [x1_bc.reshape(-1, 1), x2_bc.reshape(-1, 1)], dim=1,
    )
    return x_interior.to(device), x_boundary.to(device)


def compute_design_matrix_interior_manual(integration_weights, x, model, d):
    """PDE residual design matrix for -Δu + u (tanh; notebook)."""
    W = model.W
    b = model.b
    Z = x @ W.t() + b
    V = torch.tanh(Z)
    fprime = 1 - V ** 2
    fsecond = -2 * V * fprime
    W_spatial_sq = W[:, :d] ** 2
    laplacian_per_unit = W_spatial_sq.sum(dim=1)
    laplacian_V = fsecond * laplacian_per_unit.unsqueeze(0)
    A_in = -laplacian_V + V.detach()
    return A_in * integration_weights, V.detach()


def compute_design_matrix_boundary(x, model):
    return model.hidden(x)


def rescale_mat_rhs(A, f):
    """Row-rescale design matrix / rhs (exact notebook loop)."""
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


def make_u_exact(m1, m2, m3):
    def u_exact(x, d):
        z1 = torch.prod(torch.sin(m1 * pi * x), dim=1, keepdim=True)
        z2 = torch.prod(torch.sin(m2 * pi * x), dim=1, keepdim=True)
        z3 = torch.prod(torch.sin(m3 * pi * x), dim=1, keepdim=True)
        return z1 + z2 + z3
    return u_exact


def make_f_rhs(m1, m2, m3):
    def f_rhs(x, d):
        z1 = (d * (m1 * pi) ** 2 + 1) * torch.prod(
            torch.sin(m1 * pi * x), dim=1, keepdim=True,
        )
        z2 = (d * (m2 * pi) ** 2 + 1) * torch.prod(
            torch.sin(m2 * pi * x), dim=1, keepdim=True,
        )
        z3 = (d * (m3 * pi) ** 2 + 1) * torch.prod(
            torch.sin(m3 * pi * x), dim=1, keepdim=True,
        )
        return z1 + z2 + z3
    return f_rhs


def make_u_exact_grad(m1, m2, m3):
    def u_exact_grad(d):
        grad_list = []

        def make_u_i(i):
            def u_i(x):
                sin_terms = torch.sin(m1 * pi * x)
                cos_term = torch.cos(m1 * pi * x[:, i - 1:i])
                sin_terms[:, i - 1:i] = cos_term
                z1 = (m1 * pi) * torch.prod(sin_terms, dim=1, keepdim=True)

                sin_terms = torch.sin(m2 * pi * x)
                cos_term = torch.cos(m2 * pi * x[:, i - 1:i])
                sin_terms[:, i - 1:i] = cos_term
                z2 = (m2 * pi) * torch.prod(sin_terms, dim=1, keepdim=True)

                sin_terms = torch.sin(m3 * pi * x)
                cos_term = torch.cos(m3 * pi * x[:, i - 1:i])
                sin_terms[:, i - 1:i] = cos_term
                z3 = (m3 * pi) * torch.prod(sin_terms, dim=1, keepdim=True)
                return z1 + z2 + z3
            return u_i

        for i in range(1, d + 1):
            grad_list.append(make_u_i(i))
        return grad_list
    return u_exact_grad


def npz_path_1d():
    return DATA_DIR / f'sphere_elliptic_1d_m_{M1}_{M2}_{M3}.npz'


def npz_path_2d():
    return DATA_DIR / f'sphere_elliptic_2d_m_{M1}_{M2}_{M3}.npz'


def _eval_errors(model, u_exact, u_exact_grad_fn, d, integration_weights, pts):
    u_exact_val = u_exact(pts, d)
    diff_sqrd = (u_exact_val - model(pts)) ** 2
    err_l2 = (integration_weights.t() @ diff_sqrd) ** 0.5
    err_h1_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
    for ii, grad_i in enumerate(u_exact_grad_fn(d)):
        u_pred_grad = model.evaluate_derivative(pts, ii + 1)
        u_exact_grad_val = grad_i(pts)
        err_h1_sq = err_h1_sq + (
            integration_weights.t() @ (u_pred_grad - u_exact_grad_val) ** 2
        )
    err_h1 = err_h1_sq ** 0.5
    return float(err_l2), float(err_h1)


def run_exp_1d(
    m1=M1, m2=M2, m3=M3,
    R_m_list=None, trial_num=1, base_seed=BASE_SEED,
):
    """1D elliptic collocation (sphere); saves npz."""
    R_m_list = list(R_m_list or R_M_LIST_1D)
    d = 1
    x_l, x_r = -1.0, 1.0
    Nx_err, order_err = 1024, 5
    N_bc, N_init = 1, 4000
    neuron_num_arr = np.array([2 ** i for i in range(2, 9)], dtype=float)

    u_exact = make_u_exact(m1, m2, m3)
    f_rhs = make_f_rhs(m1, m2, m3)
    u_exact_grad_fn = make_u_exact_grad(m1, m2, m3)
    w_err, pts_err = PiecewiseGQ1D_weights_points(x_l, x_r, Nx_err, order_err)

    mean_err_l2_rows = []
    mean_err_h1_rows = []
    seed_rows = []
    seed_counter = base_seed
    for R_m in R_m_list:
        mean_err_l2_list = []
        mean_err_h1_list = []
        seed_n_list = []
        for n in neuron_num_arr.astype(int):
            err_l2_trial = torch.zeros(trial_num)
            err_h1_trial = torch.zeros(trial_num)
            seeds_trial = np.zeros(trial_num, dtype=np.int64)
            for trial in range(trial_num):
                seed = seed_counter
                seed_counter += 1
                seeds_trial[trial] = seed
                set_seed(seed)

                X_bc = generate_boundary_points(N_bc, d, x_l, x_r)
                X_init = generate_initial_points(N_init, d, x_l, x_r)
                collocation_weights = torch.ones(X_init.size(0), 1, device=device)
                f_in = f_rhs(X_init, d)
                g_bc = u_exact(X_bc, d)

                model = ELMModel(d, n, Rm=R_m).to(device)
                model.initialize_w_b_sphere(R_m)

                A_in, _ = compute_design_matrix_interior_manual(
                    collocation_weights, X_init, model, d,
                )
                A_bc = compute_design_matrix_boundary(X_bc, model)
                A_total = torch.cat([A_in, A_bc], dim=0)
                b_total = torch.cat([f_in, g_bc], dim=0)
                A_total, b_total = rescale_mat_rhs(A_total, b_total)
                # Notebook: scipy.linalg.lstsq (default gelsd)
                phi_sol = scipy.linalg.lstsq(
                    A_total.cpu().numpy(), b_total.cpu().numpy(),
                )[0]
                phi_sol = torch.from_numpy(phi_sol).to(device)
                model.phi.data.copy_(phi_sol.view(1, -1))

                err_l2, err_h1 = _eval_errors(
                    model, u_exact, u_exact_grad_fn, d, w_err, pts_err,
                )
                err_l2_trial[trial] = err_l2
                err_h1_trial[trial] = err_h1

            mean_err_l2_list.append(float(torch.mean(err_l2_trial)))
            mean_err_h1_list.append(float(torch.mean(err_h1_trial)))
            seed_n_list.append(seeds_trial)
            print(
                f'1d-elliptic-sphere m=({m1},{m2},{m3}) R={R_m} n={n}: '
                f'L2={mean_err_l2_list[-1]:.3e} H1={mean_err_h1_list[-1]:.3e}',
                flush=True,
            )
        mean_err_l2_rows.append(mean_err_l2_list)
        mean_err_h1_rows.append(mean_err_h1_list)
        seed_rows.append(seed_n_list)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = npz_path_1d()
    np.savez(
        out,
        d=1,
        m1=m1, m2=m2, m3=m3,
        Nx_err=Nx_err,
        order_err=order_err,
        N_bc=N_bc,
        N_init=N_init,
        trial_num=trial_num,
        base_seed=base_seed,
        R_m_arr=np.asarray(R_m_list, dtype=float),
        neuron_num_arr=neuron_num_arr,
        mean_err_l2=np.asarray(mean_err_l2_rows, dtype=float),
        mean_err_h1=np.asarray(mean_err_h1_rows, dtype=float),
        seed_arr=np.asarray(seed_rows, dtype=np.int64),
    )
    print(f'saved {out}', flush=True)
    return out


def run_exp_2d(
    m1=M1, m2=M2, m3=M3,
    R_m_list=None, trial_num=1, base_seed=BASE_SEED,
):
    """2D elliptic collocation (sphere); saves npz."""
    R_m_list = list(R_m_list or R_M_LIST_2D)
    d = 2
    x_l, x_r = -1.0, 1.0
    Nx_err, order_err = 50, 5
    N_grid = 100
    neuron_num_arr = np.array([2 ** i for i in range(4, 13)], dtype=float)

    u_exact = make_u_exact(m1, m2, m3)
    f_rhs = make_f_rhs(m1, m2, m3)
    u_exact_grad_fn = make_u_exact_grad(m1, m2, m3)
    w_err, pts_err = PiecewiseGQ2D_weights_points(
        Nx_err, order_err, (x_l, x_l), (x_r, x_r),
    )

    mean_err_l2_rows = []
    mean_err_h1_rows = []
    seed_rows = []
    seed_counter = base_seed
    for R_m in R_m_list:
        mean_err_l2_list = []
        mean_err_h1_list = []
        seed_n_list = []
        for n in neuron_num_arr.astype(int):
            err_l2_trial = torch.zeros(trial_num)
            err_h1_trial = torch.zeros(trial_num)
            seeds_trial = np.zeros(trial_num, dtype=np.int64)
            for trial in range(trial_num):
                seed = seed_counter
                seed_counter += 1
                seeds_trial[trial] = seed
                set_seed(seed)

                X_init, X_bc = generate_initial_points_uniform_grid(
                    N_grid, d, x_l, x_r,
                )
                collocation_weights = torch.ones(
                    X_init.size(0), 1, device=device,
                )
                f_in = f_rhs(X_init, d)
                g_bc = u_exact(X_bc, d)

                model = ELMModel(d, n, Rm=R_m).to(device)
                model.initialize_w_b_sphere(R_m)

                A_in, _ = compute_design_matrix_interior_manual(
                    collocation_weights, X_init, model, d,
                )
                A_bc = compute_design_matrix_boundary(X_bc, model)
                A_total = torch.cat([A_in, A_bc], dim=0)
                b_total = torch.cat([f_in, g_bc], dim=0)
                A_total, b_total = rescale_mat_rhs(A_total, b_total)
                # Notebook: scipy.linalg.lstsq (default gelsd)
                phi_sol = scipy.linalg.lstsq(
                    A_total.cpu().numpy(), b_total.cpu().numpy(),
                )[0]
                phi_sol = torch.from_numpy(phi_sol).to(device)
                model.phi.data.copy_(phi_sol.view(1, -1))

                err_l2, err_h1 = _eval_errors(
                    model, u_exact, u_exact_grad_fn, d, w_err, pts_err,
                )
                err_l2_trial[trial] = err_l2
                err_h1_trial[trial] = err_h1

            mean_err_l2_list.append(float(torch.mean(err_l2_trial)))
            mean_err_h1_list.append(float(torch.mean(err_h1_trial)))
            seed_n_list.append(seeds_trial)
            print(
                f'2d-elliptic-sphere m=({m1},{m2},{m3}) R={R_m} n={n}: '
                f'L2={mean_err_l2_list[-1]:.3e} H1={mean_err_h1_list[-1]:.3e}',
                flush=True,
            )
        mean_err_l2_rows.append(mean_err_l2_list)
        mean_err_h1_rows.append(mean_err_h1_list)
        seed_rows.append(seed_n_list)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = npz_path_2d()
    np.savez(
        out,
        d=2,
        m1=m1, m2=m2, m3=m3,
        Nx_err=Nx_err,
        order_err=order_err,
        N_grid=N_grid,
        trial_num=trial_num,
        base_seed=base_seed,
        R_m_arr=np.asarray(R_m_list, dtype=float),
        neuron_num_arr=neuron_num_arr,
        mean_err_l2=np.asarray(mean_err_l2_rows, dtype=float),
        mean_err_h1=np.asarray(mean_err_h1_rows, dtype=float),
        seed_arr=np.asarray(seed_rows, dtype=np.int64),
    )
    print(f'saved {out}', flush=True)
    return out


def _style_ax(ax, ylabel):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'Number of neurons $n$')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(True, which='both', ls=':', alpha=0.6)


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


def plot_panel_pair(axes, path):
    """Plot L2 (axes[0]) and H1 (axes[1]) vs neurons for each R."""
    data = np.load(path)
    x = np.asarray(data['neuron_num_arr'], dtype=float)
    R_m_arr = np.asarray(data['R_m_arr'], dtype=float)
    mean_err_l2 = np.asarray(data['mean_err_l2'], dtype=float)
    mean_err_h1 = np.asarray(data['mean_err_h1'], dtype=float)
    for i, R_m in enumerate(R_m_arr):
        label = f'radius = {int(R_m)}'
        axes[0].plot(x, mean_err_l2[i], '.-', label=label)
        axes[1].plot(x, mean_err_h1[i], '.-', label=label)
    for ax in axes:
        ax.set_xlim(x.min() / 1.15, x.max() * 1.15)
    _style_ax(axes[0], r'$L^2$ error')
    _style_ax(axes[1], r'$H^1$ error')


def plot_1d(out_name='Sphere_elliptic_1d.png'):
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=400)
    plot_panel_pair(axes, npz_path_1d())
    fig.suptitle(
        rf'Sphere scheme. $u(x)=\sum_{{i=1}}^{{3}}\sin(m_i\pi x)$, '
        rf'$m_1={M1}$, $m_2={M2}$, $m_3={M3}$',
        fontsize=12,
    )
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / out_name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}', flush=True)
    return out


def plot_2d(out_name='Sphere_elliptic_2d.png'):
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=400)
    plot_panel_pair(axes, npz_path_2d())
    fig.suptitle(
        rf'Sphere scheme. $u(x)=\sum_{{i=1}}^{{3}}\sin(m_i\pi x_1)\sin(m_i\pi x_2)$, '
        rf'$m_1={M1}$, $m_2={M2}$, $m_3={M3}$',
        fontsize=12,
    )
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / out_name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}', flush=True)
    return out


def plot_all():
    return plot_1d(), plot_2d()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='1D/2D elliptic PDE collocation (tanh sphere scheme)',
    )
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Skip experiments; replot from saved npz files',
    )
    parser.add_argument(
        '--dim', choices=['1', '2', 'all'], default='all',
        help='Which experiments to run (default: all)',
    )
    args = parser.parse_args()

    if not args.plot_only:
        if args.dim in ('1', 'all'):
            run_exp_1d(trial_num=5)
        if args.dim in ('2', 'all'):
            run_exp_2d(trial_num=5)
    if args.plot_only or args.dim == 'all':
        plot_all()
    elif args.dim == '1':
        plot_1d()
    else:
        plot_2d()
