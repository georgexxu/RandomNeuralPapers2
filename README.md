# RandomNeuralPapers2

Numerical experiments for **random feature methods (RFM / ELM)** applied to function approximation and PDEs, using shallow neural networks with fixed random (or predetermined) inner weights and a linear outer layer solved by least squares.

## Repository layout

```
RandomNeuralPapers2/
├── code/                 # Main experiments and utilities
│   ├── RFM_L2Fitting/    # L² regression / fitting
│   ├── RFM_H1Fitting/    # H¹ fitting and PDE experiments
│   ├── GPTCodebase/      # Small demo notebooks (least-squares solvers)
│   ├── utils_quad_init.py
│   ├── lstsq-solver.ipynb
│   └── RFM2d-code-todo.ipynb
├── 0figure/              # Selected paper / report figures
└── paper/                # Reference PDFs
```

> **Note:** `code/rfm_l2fitting_ly/` is present in the tree but is **not** part of the documented codebase below.

---

## `code/` — overview

| Path | Role |
|------|------|
| `utils_quad_init.py` | Shared models (`ReLU^k`, `tanh`, cosine), quadrature / Monte Carlo point generators, and weight-initialization schemes (uniform, sphere, Petrushev, Gaussian). |
| `lstsq-solver.ipynb` | Small notebook exploring least-squares solvers and singular values. |
| `RFM2d-code-todo.ipynb` | Work-in-progress 2D Poisson RFM with partition of unity (POU). |
| `GPTCodebase/` | Demo notebooks on discrete least squares with ReLU features. |
| `RFM_L2Fitting/` | L² approximation experiments (random / predetermined features). |
| `RFM_H1Fitting/` | H¹ fitting, Neumann / Poisson PINN-style RFM, Helmholtz, random vs deterministic features. |

Typical stack: **PyTorch**, NumPy, SciPy, Matplotlib (double precision by default).

---

## `code/GPTCodebase/` — least-squares demos

| File | Description |
|------|-------------|
| `l2regression1d.ipynb` | Fits a high-frequency target \(\sin(kx)\) on \([-1,1]\) with a ReLU feature dictionary (bias, linear term, and hinge pairs at knots). Compares three discrete least-squares solvers: **(1)** normal equations \((\Phi^\top\Phi)\theta=\Phi^\top y\), **(2)** QR/SVD least squares (`np.linalg.lstsq`), and **(3)** ridge / Tikhonov regularization \((\Phi^\top\Phi+\lambda I)\theta=\Phi^\top y\). Reports condition number, singular values, and MSE for each method. |

---

## `code/RFM_L2Fitting/` — L² fitting

Experiments for approximating target functions on \([-1,1]^d\) in the \(L^2\) sense with shallow random feature networks. Split by activation.

### `reluk/` — ReLU\(^k\) features

| File | Description |
|------|-------------|
| `l2regression2d-reluk.ipynb` | 2D L² regression with ReLU\(^k\). |
| `L2MinimizationPredeterminedFeature-nd.ipynb` | Multi-dimensional L² minimization with predetermined features (sphere / fixed-\(\omega\) sampling). |
| `L2MinimizationPredeterminedFeature-nd-local.ipynb` | Local / variant of the predetermined-feature experiments. |
| `L2VariationalLeastSquares.ipynb` | Variational least-squares formulation of L² fitting. |
| `L2MinimizationConditionAnalysis.ipynb` | Conditioning analysis of the L² least-squares systems. |
| `l2regression-nd.py` | Script form of multi-d L² regression. |
| `L2minimization-nd.py` | Script form of multi-d L² minimization. |
| `L2minimization-condition.py` | Script for condition-number experiments. |
| `results_relu/` | Saved `.npz` results and `plot.ipynb` for plotting. |

Target setups commonly include product sines, averaged-argument sines, and different \((\omega,b)\) sampling rules (sphere \(S^d\), fixed \(\omega\), etc.).

### `tanh/` — tanh features

| File | Description |
|------|-------------|
| `RFM-l2regression1d-tanh.ipynb` | 1D L² regression with tanh RFM. |
| `l2regression2d-tanh-abd.ipynb` | 2D tanh L² regression (ABD-style initialization). |
| `l2regression-nd-tanh-abd.ipynb` | Multi-d tanh L² regression (ABD). |
| `l2regression-nd-tanh-petrushev.ipynb` | Multi-d tanh L² regression (Petrushev-type sampling). |
| `L2FittingRFMVariational.ipynb` | Variational / quadrature-consistent L² fitting. |
| `rfm-pou-l2-tanh.ipynb` | RFM with partition of unity for 2D L² regression (`pyrfm`). |
| `data/` | Saved errors (`.pt`, `.txt`), figures, and `readData.ipynb`. |
| `data_petrushev/` | Collocation / variational result figures for Petrushev sampling. |

### Other L² notebooks

| File | Description |
|------|-------------|
| `SpecialNNL2FittingRFM.ipynb` | Special / greedy-style L² fitting with consistent quadrature. |
| `RFM5d.ipynb` | Higher-dimensional (e.g. 5D) RFM experiments with QMC loss evaluation. |

---

## `code/RFM_H1Fitting/` — H¹ fitting and PDEs

Experiments for H¹-type fitting and PDE residual least squares (PINN-style) with random or predetermined features.

| File | Description |
|------|-------------|
| `H1FittingRFM.ipynb` | Main H¹ fitting RFM notebook (NPSC-style assembly, piecewise Gauss quadrature). |
| `NeumannProblemPredeterminedFeature.ipynb` | Neumann problem on \([-1,1]^d\) with predetermined features; general dimension. |
| `neumann_problem_PINN.ipynb` | Neumann PDE via PINN least-squares loss; random fixed inner weights + direct LS. |
| `poisson_PINN.ipynb` | Poisson / related elliptic PINN-style RFM. |
| `RFM-Helmholtz1d-tanh.ipynb` | 1D Helmholtz with tanh RFM. |
| `plot_data_compare_rand.ipynb` | Plots comparing random vs deterministic feature choices. |
| `data/` | Saved H¹ error tensors and related figures; `readDatah1-1d.ipynb`. |
| `data-compare-rand/` | L² / H¹ errors and neuron lists for random vs non-random ELM (`.pt`). |
| `data-compare-rand-copy/` | Backup / copy of a subset of those result tensors. |

---

## `0figure/` and `paper/`

- **`0figure/`** — Selected figures (e.g. ReLU / tanh L² error plots, random vs deterministic PDE comparisons).
- **`paper/`** — Reference literature (e.g. FBPINN, ELM–FBPINN, RFM conditioning, Petrushev 1998).

---

## Common themes

1. **Random feature / ELM models** — Inner weights fixed (random or predetermined); outer coefficients from linear least squares.
2. **Initialization schemes** — Uniform, sphere, Petrushev, Gaussian (see `utils_quad_init.py`).
3. **Discretization** — Collocation, piecewise Gaussian quadrature, Monte Carlo / Sobol, and variational assemblies.
4. **Metrics** — Relative \(L^2\) and \(H^1\) errors, often vs neuron count and radius / scale parameter \(R_m\).
5. **Activations** — `tanh` and ReLU\(^k\) (`k = 1,2,3,\ldots`).

---

## Quick start

Most work lives in Jupyter notebooks under `code/RFM_L2Fitting/` and `code/RFM_H1Fitting/`. Shared helpers can be imported from:

```python
from utils_quad_init import model, model_tanh, initialize_w_b_sphere, PiecewiseGQ2D_weights_points
```

Run notebooks from their own directory (or ensure `code/` is on `PYTHONPATH`) so relative data paths resolve correctly.
