# Plotting condition numbers

How to regenerate the Galerkin condition-number figures from the current scripts.

## Files

| Role | Script |
|------|--------|
| L2 experiment | `L2minimization_condition.py` |
| Neumann experiment | `neumannProblem_relu_condition.py` |
| L2 plots (ReLU\(^1\), ReLU\(^2\)) | `plot_L2_condition_numbers.py` |
| Neumann plot (ReLU\(^3\)) | `plot_neumann_condition_numbers.py` |

Data and figures: `condition_number_results/`.

## Environment

```bash
conda activate pytorch
cd /path/to/linearized_nn
```

Use `MPLBACKEND=Agg` for headless plotting.

## Reproduce data (optional)

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -u L2minimization_condition.py
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -u neumannProblem_relu_condition.py
```

L2 writes `L2minimization-condition-{d}d-relu{k}.npz` for \(d=1,\ldots,6\) and \(k=1,2\).  
Neumann writes `Neumann-problem-Predetermined-Feature-{d}d-relu3.npz` for \(d=1,\ldots,6\).

### Feature sampling

| Dimension | Features | Trials |
|-----------|----------|--------|
| \(d=1\) | deterministic (`initialize_model_1d`) | 1 |
| \(d=2\) | deterministic (`initialize_model_3`) | 1 |
| \(d=3,\ldots,6\) | random sphere (`initialize_model_2`) | 5 (mean is plotted) |

### Neuron grids

- L2: requested \(N = 4,8,\ldots,512\) in all dimensions.
- Neumann: \(N = 8,\ldots,512\) for \(d\le 3\); \(N = 8,\ldots,256\) for \(d=4,5,6\).

### Keys used by the plot scripts

| Key | Role |
|-----|------|
| `actual_neuron_arr` | \(x\): kept neuron count (mean over trials when \(d\ge 3\)) |
| `condition_number_mean` | \(y\): \(\kappa\) (falls back to `condition_number_arr`) |
| `neuron_num_arr` | requested \(N\) (printed by the Neumann plot script only) |
| `condition_number_trials` | used only to print the trial count |

The plotted \(n\) is after **both** prunes (vertex redundant removal, then near-zero diagonal prune). See [How many neurons are dropped](#how-many-neurons-are-dropped).

## Reproduce figures

```bash
MPLBACKEND=Agg python -u plot_L2_condition_numbers.py
MPLBACKEND=Agg python -u plot_neumann_condition_numbers.py
```

Outputs:

- `condition_number_results/L2_condition_number_vs_neurons_relu1.png`
- `condition_number_results/L2_condition_number_vs_neurons_relu2.png`
- `condition_number_results/Neumann_condition_number_vs_neurons_relu3.png`

## What the plot scripts do

Shared by `plot_L2_condition_numbers.py` and `plot_neumann_condition_numbers.py`:

- \(2\times 3\) panels, \(d=1,\ldots,6\); panel title `$d = {d}$`; **no** figure-level title
- Log–log axes
- Solid blue curve: mean \(\kappa\) vs kept \(n\) (`.-`)
- Dashed orange curve: least-squares fit of \(\log_{10}\kappa\) vs \(\log_{10}n\); legend `fit (slope=…)`
- **No** mean\(\pm\)std band
- Drop the **first** (smallest) neuron-count point before plotting and fitting
- Axis labels: `Number of neurons $n$` and `Condition number`
- \(x\)-limits: \([n_{\min}/1.15,\, n_{\max}\times 1.15]\)
- Major \(x\)-ticks at decades only (`LogFormatterMathtext`); minor tick labels hidden
- Fonts/line widths via `rcParams` (`font.size=12`, `dpi=400`, `figsize=(12, 7)`)
- Inward ticks on all sides; dotted grid

**Y-limits differ:**

| Script | Y-limits |
|--------|----------|
| L2 | `bottom = max(y.min()/50, 1)` (matplotlib chooses the top) |
| Neumann | `ylim(max(y.min()/3, 1), y.max()*3)` so high-\(d\) panels are not empty at the top |

L2 loops over `relu_k in (1, 2)` and writes two PNGs. Neumann uses `relu_k = 3` and writes one PNG.

## Suggested paper caption ingredients

- Problem (L2 Gram vs Neumann \(H^1\)) and ReLU power
- \(d=1,2\): one deterministic dictionary; \(d\ge 3\): mean over 5 random sphere draws
- \(n\) is the number of neurons **kept** after pruning
- Smallest requested \(N\) is omitted from the figure

## How many neurons are dropped

Two pruning stages before \(\kappa\) is computed:

1. Vertex-based `remove_redundant_neuron` (inactive / fully active features on \([-1,1]^d\))
2. Near-zero diagonal prune — L2: `prune_near_zero_l2_neurons`; Neumann: `prune_near_zero_h1_neurons`  
   (`A_{ii} < 10^{-6}\times \max_j A_{jj}`)

Extra keys in the `.npz` files (not used for plotting):

| Key | Meaning |
|-----|---------|
| `neuron_num_arr` | requested \(N\) |
| `tiny_l2_trials` | L2: count removed by the near-zero L2 prune (per trial) |
| `tiny_h1_trials` | Neumann: count removed by the near-zero \(H^1\) prune (per trial) |

Total dropped \(\approx N - n_{\mathrm{kept}}\).  
The near-zero prune is stored separately; the rest is mainly vertex redundant removal.

For \(d\ge 3\), tables use the **mean over 5 trials**.

### L2 ReLU\(^1\)

| \(d\) | \(N\) | kept | dropped | near-zero L2 |
|------|------|------|---------|--------------|
| 1 | 4…512 | 3…255 | 1…257 | 0 |
| 2 | 4…512 | 4…395 | 0…117 | 0…4 |
| 3 | 4…512 | 3.6…467 | 0.4…45 | 0.2…3.2 |
| 4 | 4…512 | 4…497.8 | 0…14.2 | 0…3.2 |
| 5 | 4…512 | 4…506.6 | 0…5.4 | 0…2.8 |
| 6 | 4…512 | 4…510.8 | 0…1.2 | 0…0.8 |

Detail by \(N\) (requested → kept, dropped, tiny):

**\(d=1\):** 4→3 (−1,tiny0); 8→5 (−3); 16→9 (−7); 32→17 (−15); 64→33 (−31); 128→65 (−63); 256→129 (−127); 512→255 (−257,tiny0)

**\(d=2\):** 4→4; 8→7 (−1); 16→14 (−2); 32→28 (−4); 64→52 (−12); 128→102 (−26,tiny1); 256→201 (−55,tiny1); 512→395 (−117,tiny4)

**\(d=3\):** 4→3.6 (−0.4,tiny0.2); 8→7.4; 16→14.4; 32→30.2; 64→60.2 (−3.8); 128→119.8 (−8.2,tiny1); 256→236.6 (−19.4,tiny1); 512→467 (−45,tiny3.2)

**\(d=4\):** 4→4; 8→7.4; 16→15.6; 32→31.4; 64→63.6; 128→125.2 (−2.8,tiny1.4); 256→251.4 (−4.6,tiny1); 512→497.8 (−14.2,tiny3.2)

**\(d=5\):** 4→4; 8→8; 16→15.8; 32→31.8; 64→63.8; 128→127 (−1); 256→253.8 (−2.2,tiny0.4); 512→506.6 (−5.4,tiny2.8)

**\(d=6\):** 4→4; 8→8; 16→16; 32→32; 64→63.2 (−0.8,tiny0.4); 128→127.8; 256→255.2 (−0.8,tiny0.6); 512→510.8 (−1.2,tiny0.8)

### L2 ReLU\(^2\)

| \(d\) | \(N_{\max}\) | kept | dropped | near-zero L2 |
|------|-------------|------|---------|--------------|
| 1 | 512 | 256 | 256 | 0 |
| 2 | 512 | 387 | 125 | 15 |
| 3 | 512 | 465.4 | 46.6 | 13.4 |
| 4 | 512 | 491 | 21 | 12 |
| 5 | 512 | 502 | 10 | 7.4 |
| 6 | 512 | 508.8 | 3.2 | 2.8 |

Detail by \(N\):

**\(d=1\):** 4→4; 8→6 (−2); 16→10 (−6); 32→18 (−14); 64→34 (−30); 128→66 (−62); 256→130 (−126); 512→256 (−256,tiny0)

**\(d=2\):** 4→3 (−1,tiny1); 8→7; 16→14; 32→27 (−5,tiny1); 64→53 (−11,tiny2); 128→103 (−25,tiny3); 256→200 (−56,tiny5); 512→387 (−125,tiny15)

**\(d=3\):** 4→3.6; 8→7.2; 16→14.2; 32→30.4; 64→59.2 (−4.8,tiny1.8); 128→120.4 (−7.6,tiny2.2); 256→238.2 (−17.8,tiny4.8); 512→465.4 (−46.6,tiny13.4)

**\(d=4\):** 4→3.8; 8→7.2; 16→15.4; 32→31; 64→62 (−2,tiny2); 128→123.8 (−4.2,tiny2.8); 256→249.2 (−6.8,tiny3.2); 512→491 (−21,tiny12)

**\(d=5\):** 4→4; 8→8; 16→15.6; 32→31.2; 64→63.2; 128→125.8 (−2.2,tiny1.2); 256→252.4 (−3.6,tiny1.8); 512→502 (−10,tiny7.4)

**\(d=6\):** 4→4; 8→8; 16→16; 32→32; 64→63 (−1,tiny0.6); 128→127 (−1,tiny0.8); 256→253.6 (−2.4,tiny2.2); 512→508.8 (−3.2,tiny2.8)

### Neumann ReLU\(^3\)

| \(d\) | \(N_{\max}\) | kept | dropped | near-zero \(H^1\) |
|------|-------------|------|---------|-------------------|
| 1 | 512 | 257 | 255 | 0 |
| 2 | 512 | 391 | 121 | 15 |
| 3 | 512 | 469.6 | 42.4 | 15.2 |
| 4 | 256 | 246.6 | 9.4 | 5 |
| 5 | 256 | 251.6 | 4.4 | 3.4 |
| 6 | 256 | 253 | 3 | 2.4 |

Detail by \(N\):

**\(d=1\):** 8→7 (−1); 16→11 (−5); 32→19 (−13); 64→35 (−29); 128→67 (−61); 256→131 (−125); 512→257 (−255,tiny0)

**\(d=2\):** 8→7; 16→14; 32→27 (−5,tiny1); 64→55 (−9,tiny2); 128→107 (−21,tiny3); 256→204 (−52,tiny5); 512→391 (−121,tiny15)

**\(d=3\):** 8→7.4; 16→14.6 (−1.4,tiny1); 32→29.2; 64→58.6 (−5.4,tiny2); 128→120.6 (−7.4,tiny3.2); 256→238.2 (−17.8,tiny6.6); 512→469.6 (−42.4,tiny15.2)

**\(d=4\):** 8→7.8; 16→14.6; 32→30.6; 64→61.8; 128→120.8 (−7.2,tiny4); 256→246.6 (−9.4,tiny5)

**\(d=5\):** 8→8; 16→15.8; 32→31.4; 64→63; 128→126 (−2,tiny1.2); 256→251.6 (−4.4,tiny3.4)

**\(d=6\):** 8→8; 16→15.4; 32→31.4; 64→63.2; 128→127.2; 256→253 (−3,tiny2.4)

### Takeaway

- Most removals come from **vertex redundant** pruning, not the near-zero diagonal step.
- Dropping is **heaviest in 1D/2D** (about half of \(N\) in 1D at \(N=512\)).
- In high dimension almost all neurons are kept; the near-zero prune removes only a few (somewhat more for higher ReLU powers).
