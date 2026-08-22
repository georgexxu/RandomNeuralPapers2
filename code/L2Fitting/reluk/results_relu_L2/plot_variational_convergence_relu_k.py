"""Plot L2 minimization error vs neuron count.

Produces one figure per ReLU power (relu1 and relu2).
Reads L2minimization-{d}d-relu{k}.npz in this folder.
Each figure is a 2x3 grid for dimensions d = 1..6.

Reference line uses the theoretical rate
    optimal_rate = 1/2 + (2*relu_k + 1)/(2*d)
with a vertical scale factor (as in read-data-relu-results.ipynb)
chosen so the line sits above the empirical ReLU curve.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext

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

folder = Path(__file__).resolve().parent
m_target = 0.5

for relu_k in (1, 2):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=400)
    for ax, d in zip(axes.flat, range(1, 7)):
        path = folder / f'L2minimization-{d}d-relu{relu_k}.npz'
        data = np.load(path)
        x = np.asarray(data['actual_neuron_arr'], dtype=float)
        y = np.asarray(data['mean_err_l2_arr_2'], dtype=float)

        optimal_rate = 0.5 + (2 * relu_k + 1) / (2 * d)
        # Notebook-style vertical scale so the reference line sits above the data.
        ref_unit = y[0] * x[0] ** optimal_rate * x ** (-optimal_rate)
        scale = max(1.0, float(np.max(y / ref_unit))) * 1.5
        ref = scale * ref_unit

        ax.plot(x, ref, '-.', color='C0', label=f'optimal slope: -{optimal_rate:.2f}')
        ax.plot(x, y, '.-', color='C1', label=rf'ReLU$^{relu_k}$', linewidth=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(x.min() / 1.15, x.max() * 1.15)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if relu_k == 1:
            title = rf'$L^2$-minimization. ReLU. $\Pi_{{i=1}}^{d}\sin(m\pi x)$, $m={m_target}$'
        else:
            title = (
                rf'$L^2$-minimization. ReLU$^{relu_k}$. '
                rf'$\Pi_{{i=1}}^{d}\sin(m\pi x)$, $m={m_target}$'
            )
        ax.set_title(title)
        ax.set_xlabel(r'Number of neurons $n$')
        ax.set_ylabel(r'$L^2$ error')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(True, which='both', ls=':', alpha=0.6)

    fig.tight_layout()
    out = folder / f'L2_convergence_vs_neurons_relu{relu_k}.png'
    fig.savefig(out, bbox_inches='tight')
    print(f'saved {out}')
    plt.close(fig)
