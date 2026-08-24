import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def theory_exponent_l2(k, d):
    """Log-log slope: ||u - u_n||_{L^2} ~ n^{-1/2 - (2k+1)/(2d)}."""
    return -0.5 - (2 * k + 1) / (2 * d)


def theory_exponent_h1(k, d):
    """Log-log slope: |u - u_n|_{H^1} ~ n^{-1/2 - (2k-1)/(2d)}."""
    return -0.5 - (2 * k - 1) / (2 * d)


def load_comparison_npz(path):
    data = np.load(path)
    mc_n = data["mc_n_kept_mean"] if "mc_n_kept_mean" in data.files else data["mc_n_kept"].mean(axis=0)
    return {
        "mc_n": mc_n,
        "mc_l2": data["mc_l2_mean"],
        "mc_l2_std": data["mc_l2_std"],
        "mc_h1": data["mc_h1_mean"],
        "mc_h1_std": data["mc_h1_std"],
        "qmc_n": data["qmc_n_kept"],
        "qmc_l2": data["qmc_l2"],
        "qmc_h1": data["qmc_h1"],
    }


def plot_convergence_comparison(
    mc_n, mc_l2, mc_h1, qmc_n, qmc_l2, qmc_h1,
    k, d,
    mc_l2_std=None, mc_h1_std=None,
    label1="MC: ", label2="QMC: ", title_prefix="",
    save_path=None,
):
    exp_l2 = theory_exponent_l2(k, d)
    exp_h1 = theory_exponent_h1(k, d)
    rate_l2 = -exp_l2
    rate_h1 = -exp_h1

    def reference_line(n, error_array, rate):
        C = error_array[-1] * (n[-1] ** rate)
        return C * (n ** (-rate))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=400)
    fs_label = 18
    fs_title = 20
    fs_legend = 14
    fs_tick = 14

    for ax, mc_err, qmc_err, mc_std, ylabel, rate, exp, err_label in [
        (axes[0], mc_l2, qmc_l2, mc_l2_std, r"$L^2$ Error", rate_l2, exp_l2,
         r"$\|u-u_n\|_{L^2}$"),
        (axes[1], mc_h1, qmc_h1, mc_h1_std, r"$H^1$ Error", rate_h1, exp_h1,
         r"$|u-u_n|_{H^1}$"),
    ]:
        if mc_std is not None:
            ax.fill_between(
                mc_n,
                np.maximum(mc_err - mc_std, 1e-16),
                mc_err + mc_std,
                color="C0",
                alpha=0.25,
                label=label1 + r"mean $\pm$ std",
            )
        ax.loglog(mc_n, mc_err, "o-", color="C0", label=label1 + err_label)
        ax.loglog(qmc_n, qmc_err, "s-", color="C1", label=label2 + err_label)

        ref = reference_line(mc_n, mc_err, rate)
        ax.loglog(
            mc_n, ref, "--", color="black",
            label=fr"Ref. $n^{{{exp:.2f}}}$",
        )

        ax.set_xlabel(r"$n$", fontsize=fs_label)
        ax.set_ylabel(ylabel, fontsize=fs_label)
        ax.set_title(rf"{title_prefix}Convergence: {err_label}", fontsize=fs_title)
        ax.legend(fontsize=fs_legend)
        ax.tick_params(axis="both", labelsize=fs_tick)
        ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "results_relu_qmc_compare"
    out_dir.mkdir(exist_ok=True)
    k = 3

    d3 = load_comparison_npz(out_dir / "mqc_compare_trials.npz")
    plot_convergence_comparison(
        d3["mc_n"], d3["mc_l2"], d3["mc_h1"],
        d3["qmc_n"], d3["qmc_l2"], d3["qmc_h1"],
        k=k, d=3,
        mc_l2_std=d3["mc_l2_std"],
        mc_h1_std=d3["mc_h1_std"],
        title_prefix="3D ",
        save_path=out_dir / "3dneumann-qmc-compare.png",
    )

    d5 = load_comparison_npz(out_dir / "mqc_compare_trials_5d.npz")
    plot_convergence_comparison(
        d5["mc_n"][1:], d5["mc_l2"][1:], d5["mc_h1"][1:],
        d5["qmc_n"][1:], d5["qmc_l2"][1:], d5["qmc_h1"][1:],
        k=k, d=5,
        mc_l2_std=d5["mc_l2_std"][1:],
        mc_h1_std=d5["mc_h1_std"][1:],
        title_prefix="5D ",
        save_path=out_dir / "5d-neumann-qmc-compare.png",
    )
