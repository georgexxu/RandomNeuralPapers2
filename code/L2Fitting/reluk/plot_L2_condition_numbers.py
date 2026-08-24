"""Plot L2 Gram condition number vs kept neuron count.

Produces one figure per ReLU power (relu1 and relu2).
Reads condition_number_relu/L2minimization-condition-{d}d-relu{k}.npz.
Plots mean κ curve and log-log fit only (no ±std band).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 1.1,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})

folder = Path(__file__).resolve().parent / 'condition_number_relu'

for relu_k in (1, 2):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=400)
    for ax, d in zip(axes.flat, range(1, 7)):
        path = folder / f'L2minimization-condition-{d}d-relu{relu_k}.npz'
        data = np.load(path)
        x = np.asarray(data['actual_neuron_arr'], dtype=float)
        y = np.asarray(
            data['condition_number_mean'] if 'condition_number_mean' in data.files
            else data['condition_number_arr'],
            dtype=float,
        )

        # drop the first (smallest) neuron count
        x, y = x[1:], y[1:]

        ax.plot(x, y, '.-', color='C0', label=rf'ReLU$^{relu_k}$')

        m, b = np.polyfit(np.log10(x), np.log10(y), 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = 10 ** (m * np.log10(x_fit) + b)
        ax.plot(x_fit, y_fit, '--', color='C1', label=f'fit (slope={m:.1f})')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(x.min() / 1.15, x.max() * 1.15)
        ax.set_ylim(bottom=max(y.min() / 50, 1e0))
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(rf'$d = {d}$')
        ax.set_xlabel(r'Number of neurons $n$')
        ax.set_ylabel('Condition number')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(True, which='both', ls=':', alpha=0.6)

    fig.tight_layout()
    out = folder / f'L2_condition_number_vs_neurons_relu{relu_k}.png'
    fig.savefig(out, bbox_inches='tight')
    print(f'saved {out}')
    plt.close(fig)
