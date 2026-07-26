# RandomNeuralPapers2
This git repo contains code for the paper Solving High-Dimensional PDEs Using Linearized Neural Networks https://arxiv.org/abs/2601.11771

Numerical experiments for **random feature methods (RFM / ELM)** applied to function approximation and PDEs: shallow networks with fixed random or predetermined inner weights, and a linear outer layer solved by least squares / variational assembly.

## Repository layout

```
RandomNeuralPapers2/
├── code/                 # Active experiments and utilities
│   ├── RFM_L2Fitting/    # L² regression / fitting
│   ├── RFM_H1Fitting/    # H¹ / Neumann / PINN-style PDE experiments
│   ├── GPTCodebase/      # Small least-squares demos
│   └── utils_quad_init.py
├── archived/             # Older / unused notebooks and code
├── 0figure/              # Selected paper / report figures
└── paper/                # Reference PDFs
```

Typical stack: **PyTorch**, NumPy, SciPy, Matplotlib (double precision by default).

---

## `code/` — overview

| Path | Role |
|------|------|
| `utils_quad_init.py` | Shared models (`ReLU^k`, `tanh`, cosine), quadrature / Monte Carlo generators, and weight init (uniform, sphere, Petrushev, Gaussian). |
| `GPTCodebase/` | Discrete least-squares demos and solver diagnostics. |
| `RFM_L2Fitting/` | \(L^2\) approximation with ReLU\(^k\) and tanh features. |
| `RFM_H1Fitting/` | Weak-form Neumann RFM and strong-form PINN / collocation experiments. |

---

## `code/GPTCodebase/` — least-squares demos

| File | Description |
|------|-------------|
| `l2regression1d.ipynb` | Fit \(\sin(kx)\) on \([-1,1]\) with a ReLU hinge dictionary. Compares **normal equations**, **QR/SVD least squares**, and **ridge / Tikhonov**. |
| `lstsq-solver.ipynb` | Small notebook on least-squares solvers and singular values. |

---

## `code/RFM_L2Fitting/` — L² fitting

Approximate targets on \([-1,1]^d\) in the \(L^2\) sense with shallow random / predetermined feature networks.

### `reluk/` — ReLU\(^k\)

| File | Description |
|------|-------------|
| `l2regression2d-reluk.ipynb` | 2D \(L^2\) regression with ReLU\(^k\). |
| `L2MinimizationPredeterminedFeature-nd.ipynb` | Multi-d \(L^2\) minimization with predetermined features (sphere / fixed-\(\omega\) sampling). |
| `L2VariationalLeastSquares.ipynb` | Variational least-squares formulation of \(L^2\) fitting. |
| `L2MinimizationConditionAnalysis.ipynb` | Conditioning of the \(L^2\) least-squares systems. |
| `l2regression-nd.py` | Script form of multi-d \(L^2\) regression. |
| `L2minimization-nd.py` | Script form of multi-d \(L^2\) minimization. |
| `L2minimization-condition.py` | Script for condition-number experiments. |
| `results_relu/` | Saved `.npz` results and `plot.ipynb` (variational LS vs mass-matrix assembly). |

Common targets: product sines, averaged-argument sines; sampling on \(S^d\) or with fixed \(\omega\).

### `tanh/` — tanh

| File | Description |
|------|-------------|
| `L2FittingRFMVariational.ipynb` | Variational / quadrature-consistent \(L^2\) fitting with tanh RFM. |
| `l2regression-nd-tanh-petrushev.ipynb` | Multi-d tanh \(L^2\) regression with Petrushev-type sampling. |
| `data/` | Saved errors, figures, and related read-out notebooks. |
| `data_petrushev/` | Collocation / variational result figures for Petrushev sampling. |

---

## `code/RFM_H1Fitting/` — H¹ / PDE experiments

Elliptic model problem \(-\operatorname{div}(\alpha\nabla u)+u=f\) (or strong form \(-\Delta u+u=f\)) with random or predetermined features.

| File | Description |
|------|-------------|
| `neumanProblemVariational_RFM.ipynb` | Weak / \(H^1\) (variational) RFM for the Neumann problem; primarily **tanh** features with random init (uniform / sphere / Petrushev). Galerkin-style mass+stiffness assembly and an \(H^1\) least-squares variant. |
| `neumannProblemPredeterminedFeature_relu.ipynb` | Same weak Neumann form with **predetermined ReLU\(^k\)** features (structured / sphere init, redundant-neuron removal); general dimension. |
| `neumannProblem_PINN.ipynb` | Strong-form **PINN / collocation** least squares with random ELM features (tanh or ReLU). Interior residual \(-\Delta u+u-f\); BCs enforced as **Dirichlet** collocation (despite the filename). |
| `plot_data_compare_rand.ipynb` | Plots comparing random vs deterministic / non-random feature choices. |
| `data/` | Saved H¹ error tensors and figures; `readDatah1-1d.ipynb`. |
| `data-compare-rand/` | \(L^2\) / \(H^1\) errors and neuron lists for random vs non-random ELM (`.pt`). |

**How the three main solvers differ**

| Notebook | Form | BC | Features |
|----------|------|----|----------|
| `neumanProblemVariational_RFM` | Weak \(H^1\) | Neumann \(g_N\) | Random tanh (ELM-style) |
| `neumannProblemPredeterminedFeature_relu` | Weak \(H^1\) | Neumann \(g_N\) | Predetermined ReLU\(^k\) |
| `neumannProblem_PINN` | Strong residual (PINN) | Dirichlet collocation | Random ELM (tanh / ReLU) |

---

## `archived/`

Older or unused notebooks and code kept for reference (e.g. Helmholtz, POU experiments, ABD tanh runs, `poisson_PINN`, local predetermined-feature copies, WIP RFM-2D, and `rfm_l2fitting_ly/`). Not part of the active experiment path above.

---

## `0figure/` and `paper/`

- **`0figure/`** — Selected figures (ReLU / tanh \(L^2\) plots, random vs deterministic PDE comparisons).
- **`paper/`** — Reference literature (FBPINN, ELM–FBPINN, RFM conditioning, Petrushev 1998).

---

## Common themes

1. **Random feature / ELM models** — Inner weights fixed (random or predetermined); outer layer from linear LS or variational assembly.
2. **Initialization** — Uniform, sphere, Petrushev, Gaussian (`utils_quad_init.py`).
3. **Discretization** — Collocation, piecewise Gauss quadrature, Monte Carlo / Sobol, variational forms.
4. **Metrics** — Relative \(L^2\) and \(H^1\) errors vs neuron count and scale \(R_m\).
5. **Activations** — `tanh` and ReLU\(^k\) (\(k=1,2,3,\ldots\)).

---

## Quick start

Most work lives under `code/RFM_L2Fitting/` and `code/RFM_H1Fitting/`. Shared helpers:

```python
from utils_quad_init import model, model_tanh, initialize_w_b_sphere, PiecewiseGQ2D_weights_points
```

Run notebooks from their own directory (or put `code/` on `PYTHONPATH`) so relative data paths resolve.
