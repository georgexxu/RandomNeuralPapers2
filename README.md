# RandomNeuralPapers2

Code for [Solving High-Dimensional PDEs Using Linearized Neural Networks](https://arxiv.org/abs/2601.11771).

Numerical experiments for **linearized / random-feature** shallow networks: inner weights are fixed (random or predetermined), and the outer layer is obtained by a linear solve (variational assembly or collocation least squares).

## Setup

Typical stack: **PyTorch**, NumPy, SciPy, Matplotlib (double precision). The ReLU scripts also use **SymPy**. GPU is used if `torch.cuda.is_available()`.

Activate the torch environment used for the paper, then run scripts from the directory that contains them (relative data paths). Shared helpers live in `code/utils_quad_init.py`; the tanh Python scripts add `code/` to `sys.path` automatically.

```bash
# example
conda activate pytorch   # or your torch env
cd code/L2Fitting/tanh
python L2minimizationVariational1d2dRFM.py
```

To replot from saved `.npz` files without rerunning experiments, pass `--plot-only` where that flag exists (tanh L2 scripts and `tanh_ellipticProblem1d2d_sphere.py`).

Paper numbering used below matches `reluk_numerical_paper/main.tex` §4:

- **§4.1** ReLU\(^k\) networks (L², collocation, condition numbers, Neumann PDE, QMC)
- **§4.2** tanh networks (L², collocation, deterministic Petrushev / sphere schemes, elliptic collocation)

---

## Repository layout

```
RandomNeuralPapers2/
├── code/
│   ├── utils_quad_init.py   # models, quadrature / MC, weight init
│   ├── L2Fitting/
│   │   ├── tanh/            # §4.2 tanh L² experiments
│   │   └── reluk/           # §4.1 ReLU^k L² experiments
│   └── H1Fitting/           # §4.1.4 Neumann + QMC; §4.2 tanh elliptic PDE
├── archived/                # older notebooks, not the reproduction path
└── reluk_numerical_paper/   # paper source
```

Jupyter notebooks next to the scripts are the original research / development code (exploratory implementations, intermediate runs, parameter studies). The standalone `.py` files are the consolidated reproduction path.

---

## `code/L2Fitting/tanh/` — tanh L² minimization (§4.2)

In a working torch environment these scripts run as-is (1D and 2D). Results and figures are written to `data/`.

| File | Role |
|------|------|
| `L2minimizationVariational1d2dRFM.py` | Variational formulation; **random** (uniform) sampling of nonlinear parameters |
| `l2regression1d2dRFM.py` | Collocation formulation; **random** sampling of nonlinear parameters |
| `l2regression1d2dPetrushev.py` | Collocation; **Petrushev** scheme |
| `l2regression1d2dSphere.py` | Collocation; **sphere** scheme |

```bash
cd code/L2Fitting/tanh
python L2minimizationVariational1d2dRFM.py
python l2regression1d2dRFM.py
python l2regression1d2dPetrushev.py
python l2regression1d2dSphere.py

# figures only, from saved npz
python l2regression1d2dRFM.py --plot-only
```

Related notebooks: `L2FittingRFMVariational.ipynb`, `l2regression-nd-tanh-petrushev.ipynb`.

---

## `code/L2Fitting/reluk/` — ReLU\(^k\) L² minimization (§4.1)

| File | Role |
|------|------|
| `L2minimization-nd.py` | Variational L² minimization, \(d = 1,\ldots,6\) |
| `l2regression-nd.py` | Collocation L² minimization, \(d = 1,\ldots,6\) |
| `L2minimization_condition.py` | Condition number of the mass matrix (variational form), \(d = 1,\ldots,6\) |

**How to set \(k\).** In `L2minimization-nd.py` and `l2regression-nd.py`, set every `relu_k = ...` assignment to `1` or `2` (the default in the file is `2`). Each script then runs **all dimensions \(d=1\)–\(6\) sequentially** in one process.

`L2minimization_condition.py` already loops `relu_k in (1, 2)` and `d = 1..6`; you do not need to edit \(k\) there.

```bash
cd code/L2Fitting/reluk
python L2minimization-nd.py          # after setting relu_k
python l2regression-nd.py            # after setting relu_k
python L2minimization_condition.py   # k=1 and k=2, all d
```

If you re-run the first two scripts, they currently write `.npz` files to a local `results_relu/` folder. The paper plots use the committed data in `results_relu_L2/`. Copy or move the new files into `results_relu_L2/` before plotting, or keep using the committed files.

### Plot error decay (§4.1.1–4.1.2)

Data: `results_relu_L2/`. From that folder:

```bash
cd code/L2Fitting/reluk/results_relu_L2
python plot_variational_convergence_relu_k.py
python plot_collocation_convergence_relu_k.py
```

### Plot L² condition numbers (§4.1.3)

Committed data: `condition_number_relu/`. (A fresh run of `L2minimization_condition.py` writes `condition_number_results/` unless you change `folder` in that script.)

```bash
cd code/L2Fitting/reluk
python plot_L2_condition_numbers.py
```

Related notebooks: `L2MinimizationPredeterminedFeature-nd.ipynb`, `L2VariationalLeastSquares.ipynb`, `l2regression2d-reluk.ipynb`.

---

## `code/H1Fitting/` — PDEs

### ReLU\(^k\) Neumann problem (§4.1.4)

| File | Role |
|------|------|
| `neumannProblemPredeterminedFeature_relu.ipynb` | Reproduce Neumann results for \(d = 2,\ldots,6\) (predetermined ReLU\(^k\) features, variational form). Run the cells for the dimension you want. |
| `neumannProblem_relu_condition.py` | Condition numbers for the Neumann problem with **ReLU\(^3\)**, \(d = 1,\ldots,6\) |

```bash
cd code/H1Fitting
python neumannProblem_relu_condition.py
```

Committed condition-number data: `condition_number_neumann/`. Plot with:

```bash
cd code/H1Fitting
python plot_neumann_condition_numbers.py
```

A fresh run of `neumannProblem_relu_condition.py` writes `condition_number_results/` unless you change `folder` in that script. Copy files into `condition_number_neumann/` to match the plot script, or point `folder` at `condition_number_neumann/`.

(`neumannProblemPredeterminedFeature_relu_condition.py` is an older, cell-style version of the same experiment.)

### QMC vs MC for nonlinear parameters (§4.1.5)

| File | Role |
|------|------|
| `neumannProblem3d_QMC_compare.py` | 3D Neumann, MC vs QMC feature parameters |
| `neumannProblem5d_QMC_compare.py` | 5D Neumann, MC vs QMC |
| `plot_qmc_compare.py` | Figures from saved data |

Data: `results_relu_qmc_compare/`. To plot without rerunning:

```bash
cd code/H1Fitting
python plot_qmc_compare.py
```

Rerunning the compare scripts writes `.npz` into `results_relu_qmc_compare/` (run them from `code/H1Fitting/`).

### Tanh elliptic PDE, collocation, sphere scheme (§4.2)

| File | Role |
|------|------|
| `tanh_ellipticProblem1d2d_sphere.py` | Elliptic PDE in 1D and 2D, collocation, sphere scheme |

```bash
cd code/H1Fitting
python tanh_ellipticProblem1d2d_sphere.py
python tanh_ellipticProblem1d2d_sphere.py --plot-only
python tanh_ellipticProblem1d2d_sphere.py --dim 1
```

Related notebooks: `neumanProblemVariational_RFM.ipynb` (tanh variational Neumann / RFM), `tanh_ellipticProblem_collocation_sphere.ipynb`.

---

## `archived/`

Older or unused notebooks (Helmholtz, POU, PINN copies, GPT least-squares demos, etc.). Not needed to reproduce the paper figures above.
