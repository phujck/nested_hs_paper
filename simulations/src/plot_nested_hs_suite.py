import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suite_common import ensure_dir


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def plot_cg(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cg_nhs_cg_1.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for mode in np.unique(arr["mode_set"]):
        sub = arr[arr["mode_set"] == mode]
        max_cut = np.max(sub["cutoff"])
        sub = sub[sub["cutoff"] == max_cut]
        order = np.argsort(sub["beta_lambda"])
        sub = sub[order]

        axes[0].plot(sub["beta_lambda"], sub["p0_pred"], linewidth=1.8, label=f"pred mode {mode}")
        axes[0].plot(sub["beta_lambda"], sub["p0_exact"], marker="o", linestyle="None", label=f"ED mode {mode}")

    axes[0].set_xlabel(r"$\beta\lambda$")
    axes[0].set_ylabel(r"$p_0$")
    axes[0].set_title("NHS-CG-1: ED vs commuting analytic prediction")
    axes[0].legend(fontsize=8)

    for mode in np.unique(arr["mode_set"]):
        subm = arr[arr["mode_set"] == mode]
        for beta in np.unique(subm["beta"]):
            sub = subm[subm["beta"] == beta]
            order = np.argsort(sub["cutoff"])
            sub = sub[order]
            axes[1].plot(sub["cutoff"], sub["trace_distance"], marker="o", linestyle="-", label=f"{mode}, beta={beta}")

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Cutoff")
    axes[1].set_ylabel("Trace distance")
    axes[1].set_title("NHS-CG-1: residual convergence")
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_cg_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "nhs_cg_1.png"), dpi=150)
    plt.close(fig)


def plot_cng(data_dir, fig_dir):
    summary = read_csv(os.path.join(data_dir, "cng_nhs_cng_1.csv"))
    curves = read_csv(os.path.join(data_dir, "cng_nhs_cng_1_curves.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: PSD ED vs MC overlay at largest N.
    n_max = int(np.max(curves["N"]))
    sub = curves[(curves["N"] == n_max) & (curves["branch"] == "PSD")]
    order = np.argsort(sub["m"])
    sub = sub[order]
    axes[0].plot(sub["m"], sub["p_exact"], "o-", label="ED/analytic")
    axes[0].plot(sub["m"], sub["p_mc"], "s--", label="HS MC")
    axes[0].set_xlabel("Magnetization sector m")
    axes[0].set_ylabel("Probability")
    axes[0].set_title(f"NHS-CNG-1 PSD overlay (N={n_max})")
    axes[0].legend(fontsize=8)

    # Panel 2: non-PSD ED vs MC overlay.
    sub = curves[(curves["N"] == n_max) & (curves["branch"] == "nonPSD")]
    order = np.argsort(sub["m"])
    sub = sub[order]
    axes[1].plot(sub["m"], sub["p_exact"], "o-", label="ED/analytic")
    axes[1].plot(sub["m"], sub["p_mc"], "s--", label="HS contour")
    axes[1].set_xlabel("Magnetization sector m")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"NHS-CNG-1 non-PSD overlay (N={n_max})")
    axes[1].legend(fontsize=8)

    # Panel 3: residual scaling and leakage.
    sub = summary[summary["N"] == n_max]
    for branch, marker in (("PSD", "o"), ("nonPSD", "s")):
        sb = sub[sub["branch"] == branch]
        order = np.argsort(sb["M"])
        sb = sb[order]
        axes[2].plot(sb["M"], sb["trace_distance"], marker=marker, linestyle="-", label=f"TD {branch}")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("M")
    axes[2].set_ylabel("Trace distance")
    axes[2].set_title(f"NHS-CNG-1 residual scaling (N={n_max})")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_cng_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "nhs_cng_1.png"), dpi=150)
    plt.close(fig)


def plot_ncg(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "ncg_nhs_ncg_1.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for beta in np.unique(arr["beta"]):
        sub = arr[arr["beta"] == beta]
        order = np.argsort(sub["coupling"])
        sub = sub[order]
        axes[0].plot(sub["coupling"], sub["p0_pred"], linestyle="-", label=fr"pred, $\beta={beta}$")
        axes[0].plot(sub["coupling"], sub["p0_exact"], marker="o", linestyle="None", label=fr"ED, $\beta={beta}$")

    axes[0].set_xlabel("Coupling")
    axes[0].set_ylabel(r"$p_0$")
    axes[0].set_title("NHS-NCG-1: ED vs Gaussian prediction")
    axes[0].legend(fontsize=7, ncol=2)

    for beta in np.unique(arr["beta"]):
        sub = arr[arr["beta"] == beta]
        order = np.argsort(sub["coupling"])
        sub = sub[order]
        axes[1].plot(sub["coupling"], sub["trace_distance_pred"], marker="o", linestyle="-", label=fr"$\beta={beta}$")

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Coupling")
    axes[1].set_ylabel("Trace distance")
    axes[1].set_title("NHS-NCG-1: residual")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_ncg_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "nhs_ncg_1.png"), dpi=150)
    plt.close(fig)


def plot_ncng(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "ncng_nhs_ncng_1.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    beta0 = np.median(np.unique(arr["beta"]))
    template0 = np.min(arr["template"])
    sub = arr[(np.isclose(arr["beta"], beta0)) & (arr["template"] == template0)]
    order = np.argsort(sub["coupling"])
    sub = sub[order]

    axes[0].plot(sub["coupling"], sub["p0_exact"], "o-", label=r"ED $p_0$")
    axes[0].plot(sub["coupling"], sub["p0_k2"], "s--", label=r"K2 $p_0$")
    axes[0].plot(sub["coupling"], sub["p0_k2k4"], "^-", label=r"K2+K4 $p_0$")
    axes[0].set_xlabel("Coupling")
    axes[0].set_ylabel(r"$p_0$")
    axes[0].set_title(f"NHS-NCNG-1 overlay (beta={beta0:.2f})")
    axes[0].legend(fontsize=8)

    sc = axes[1].scatter(arr["beta"], arr["coupling"], c=arr["improvement_ratio"], cmap="viridis")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel("Coupling")
    axes[1].set_title("NHS-NCNG-1 improvement map")
    fig.colorbar(sc, ax=axes[1], label=r"$D_{K2}/D_{K2+K4}$")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_ncng_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "nhs_ncng_1.png"), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot nested HS suite")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    plot_cg(data_dir, fig_dir)
    plot_cng(data_dir, fig_dir)
    plot_ncg(data_dir, fig_dir)
    plot_ncng(data_dir, fig_dir)
    print("nested_hs figures generated")


if __name__ == "__main__":
    main()
