import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suite_common_v2 import ensure_dir


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def _sort_unique(values):
    vals = np.unique(values)
    return np.sort(vals.astype(float))


def _closest(values, target):
    values = np.asarray(values, dtype=float)
    return float(values[np.argmin(np.abs(values - float(target)))])


def plot_cg(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cg_nhs_cg_1_v2.csv"))
    bench = arr[arr["record_type"] == "benchmark"]
    scaling = arr[arr["record_type"] == "mc_scaling"]

    max_cutoff = int(np.max(bench["cutoff"]))
    bench_max = bench[bench["cutoff"] == max_cutoff]
    beta_ref = _closest(bench_max["beta"], np.median(_sort_unique(bench_max["beta"])))

    couplings = _sort_unique(bench_max["coupling"])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(couplings)))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    # Panel A: exact ED + analytic + HS overlays for coupled level p2(beta).
    for color, g in zip(colors, couplings):
        sub = bench_max[np.isclose(bench_max["coupling"], g)]
        sub = np.sort(sub, order="beta")
        axes[0].plot(sub["beta"], sub["p2_analytic"], color=color, linewidth=2.0, label=fr"$g={g:.2f}$")
        axes[0].plot(sub["beta"], sub["p2_ed"], "o", color=color, markersize=4.8)
        axes[0].plot(sub["beta"], sub["p2_hs_scalar"], "s", color=color, markersize=4.4, fillstyle="none")
        axes[0].plot(sub["beta"], sub["p2_hs_path"], "^", color=color, markersize=4.8, fillstyle="none")

    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$p_2$")
    axes[0].set_title(f"(A) Coupled-level population, cutoff={max_cutoff}")

    method_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black", label="ED"),
        plt.Line2D([0], [0], linestyle="-", color="black", label="analytic"),
        plt.Line2D([0], [0], marker="s", linestyle="None", color="black", fillstyle="none", label="HS-scalar"),
        plt.Line2D([0], [0], marker="^", linestyle="None", color="black", fillstyle="none", label="HS-path"),
    ]
    leg1 = axes[0].legend(handles=method_handles, loc="upper right", fontsize=8)
    axes[0].add_artist(leg1)
    axes[0].legend(loc="lower left", fontsize=8, title="Coupling")

    # Panel B: cutoff convergence + MC scaling slope diagnostics.
    for color, g in zip(colors, couplings):
        sub = bench[np.isclose(bench["coupling"], g) & np.isclose(bench["beta"], beta_ref)]
        sub = np.sort(sub, order="cutoff")
        axes[1].plot(sub["cutoff"], sub["td_ed_analytic"], "o-", color=color, linewidth=1.8, markersize=4.5)
        # reference HS residuals at largest cutoff for same beta/g
        top = sub[sub["cutoff"] == np.max(sub["cutoff"])][0]
        axes[1].hlines(float(top["td_hs_scalar_analytic"]), xmin=np.min(sub["cutoff"]), xmax=np.max(sub["cutoff"]), color=color, linestyle="--", linewidth=1.1)
        axes[1].hlines(float(top["td_hs_path_analytic"]), xmin=np.min(sub["cutoff"]), xmax=np.max(sub["cutoff"]), color=color, linestyle=":", linewidth=1.3)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Bosonic cutoff")
    axes[1].set_ylabel("Trace distance to analytic")
    axes[1].set_title(rf"(B) Cutoff convergence at $\beta={beta_ref:.2f}$")
    style_handles = [
        plt.Line2D([0], [0], linestyle="-", color="black", label="ED"),
        plt.Line2D([0], [0], linestyle="--", color="black", label="HS-scalar"),
        plt.Line2D([0], [0], linestyle=":", color="black", label="HS-path"),
    ]
    axes[1].legend(handles=style_handles, loc="upper right", fontsize=8)

    if len(scaling) > 3:
        inset = axes[1].inset_axes([0.10, 0.08, 0.46, 0.40])
        for method, marker in (("hs_scalar", "s"), ("hs_path", "^")):
            sub = scaling[scaling["mc_method"] == method]
            if len(sub) < 2:
                continue
            sub = np.sort(sub, order="mc_M")
            inset.loglog(sub["mc_M"], sub["mc_error_to_analytic"], marker + "-", label=method.replace("_", "-"))
        inset.set_xlabel(r"$M$", fontsize=7)
        inset.set_ylabel("error", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.legend(fontsize=6, frameon=False)
        inset.set_title(r"$1/\sqrt{M}$ check", fontsize=7)

    fig.suptitle("NHS-CG-1-v2: commuting Gaussian qutrit benchmark", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_cg_1_v2.pdf"), dpi=170)
    fig.savefig(os.path.join(fig_dir, "nhs_cg_1_v2.png"), dpi=170)
    plt.close(fig)


def plot_synth_nonpsd(data_dir, fig_dir):
    summary = read_csv(os.path.join(data_dir, "cng_nhs_synth_nonpsd_v2.csv"))
    curves = read_csv(os.path.join(data_dir, "cng_nhs_synth_nonpsd_curves_v2.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    for ax, branch, title in (
        (axes[0], "PSD", r"(A) PSD branch (multi-qubit sectors)"),
        (axes[1], "nonPSD", r"(B) non-PSD branch (contour rotated)"),
    ):
        sub = curves[curves["branch"] == branch]
        sub = np.sort(sub, order="m")
        mvals = np.asarray(sub["m"], dtype=float)
        p_an = np.asarray(sub["p_analytic"], dtype=float)
        p_mc = np.asarray(sub["p_mc"], dtype=float)
        width = 0.35
        x = np.arange(len(mvals))
        ax.bar(x - width / 2.0, p_an, width=width, label="analytic")
        ax.bar(x + width / 2.0, p_mc, width=width, label="MC")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in mvals])
        ax.set_xlabel(r"magnetization sector $m$")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend(fontsize=8)

    for branch, marker in (("PSD", "o"), ("nonPSD", "s")):
        sub = summary[summary["branch"] == branch]
        sub = np.sort(sub, order="mc_M")
        axes[2].plot(sub["mc_M"], sub["trace_distance"], marker + "-", label=branch)
    if len(summary) > 1:
        s0 = np.sort(summary[summary["branch"] == "PSD"], order="mc_M")
        if len(s0) > 1:
            m0 = float(s0["mc_M"][0])
            td0 = float(s0["trace_distance"][0])
            scale = td0 * np.sqrt(max(m0, 1.0))
            marr = np.asarray(s0["mc_M"], dtype=float)
            axes[2].plot(marr, scale / np.sqrt(np.maximum(marr, 1.0)), "--", color="gray", label=r"$\propto 1/\sqrt{M}$")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$M$")
    axes[2].set_ylabel("Trace distance")
    axes[2].set_title("(C) MC convergence")
    axes[2].legend(fontsize=8)

    fig.suptitle("NHS-CNG-MAG-v2: multi-qubit magnetization benchmark", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_synth_nonpsd_v2.pdf"), dpi=170)
    fig.savefig(os.path.join(fig_dir, "nhs_synth_nonpsd_v2.png"), dpi=170)
    fig.savefig(os.path.join(fig_dir, "nhs_multiqubit_mag_v2.pdf"), dpi=170)
    fig.savefig(os.path.join(fig_dir, "nhs_multiqubit_mag_v2.png"), dpi=170)
    plt.close(fig)


def plot_phase_clock(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cng_nhs_phase_clock_v2.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    tau_vals = _sort_unique(arr["tau_c"])
    colors = ["#e63946", "#457b9d", "#2a9d8f"]
    tau_ref = tau_vals[len(tau_vals) // 2]
    has_std = "trace_distance_std" in arr.dtype.names and "p0_mc_std" in arr.dtype.names

    # Analytic curves are tau-independent in this commuting phase benchmark.
    a0 = np.sort(arr[np.isclose(arr["tau_c"], tau_ref)], order="phi_over_pi")
    axes[0].plot(a0["phi_over_pi"], a0["p0_analytic"], color=colors[0], linewidth=2.0, label=r"$p_0$ analytic")
    axes[0].plot(a0["phi_over_pi"], a0["p1_analytic"], color=colors[1], linewidth=2.0, label=r"$p_1$ analytic")
    axes[0].plot(a0["phi_over_pi"], a0["p2_analytic"], color=colors[2], linewidth=2.0, label=r"$p_2$ analytic")

    sub_ref = np.sort(arr[np.isclose(arr["tau_c"], tau_ref)], order="phi_over_pi")
    marker_kwargs = dict(marker="o", linestyle="None", markersize=3.8, alpha=0.85)
    if has_std:
        axes[0].errorbar(sub_ref["phi_over_pi"], sub_ref["p0_mc"], yerr=sub_ref["p0_mc_std"], color=colors[0], capsize=1.8, **marker_kwargs)
        axes[0].errorbar(sub_ref["phi_over_pi"], sub_ref["p1_mc"], yerr=sub_ref["p1_mc_std"], color=colors[1], capsize=1.8, **marker_kwargs)
        axes[0].errorbar(sub_ref["phi_over_pi"], sub_ref["p2_mc"], yerr=sub_ref["p2_mc_std"], color=colors[2], capsize=1.8, **marker_kwargs)
    else:
        axes[0].plot(sub_ref["phi_over_pi"], sub_ref["p0_mc"], color=colors[0], **marker_kwargs)
        axes[0].plot(sub_ref["phi_over_pi"], sub_ref["p1_mc"], color=colors[1], **marker_kwargs)
        axes[0].plot(sub_ref["phi_over_pi"], sub_ref["p2_mc"], color=colors[2], **marker_kwargs)

    legend_pop = [
        plt.Line2D([0], [0], linestyle="-", color=colors[0], label=r"$p_0$"),
        plt.Line2D([0], [0], linestyle="-", color=colors[1], label=r"$p_1$"),
        plt.Line2D([0], [0], linestyle="-", color=colors[2], label=r"$p_2$"),
    ]
    legend_tau = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black", label=fr"MC at $\tau_c={tau_ref:.2f}$"),
    ]
    leg = axes[0].legend(handles=legend_pop, loc="upper right", fontsize=8)
    axes[0].add_artist(leg)
    axes[0].legend(handles=legend_tau, loc="lower center", fontsize=8)
    axes[0].set_xlabel(r"$\phi/\pi$")
    axes[0].set_ylabel("Population")
    axes[0].set_title("(A) Smooth analytic curves + MC overlays")

    markers = ["o", "s", "^"]
    for marker, tau_c in zip(markers, tau_vals):
        sub = np.sort(arr[np.isclose(arr["tau_c"], tau_c)], order="phi_over_pi")
        x = np.asarray(sub["phi_over_pi"], dtype=float)
        y = np.asarray(sub["trace_distance"], dtype=float)
        axes[1].plot(x, y, marker=marker, linestyle="-", linewidth=1.4, markersize=3.6, label=fr"$\tau_c={tau_c:.2f}$")
        if has_std:
            ystd = np.asarray(sub["trace_distance_std"], dtype=float)
            lo = np.maximum(y - ystd, 0.0)
            hi = y + ystd
            axes[1].fill_between(x, lo, hi, alpha=0.16)
    axes[1].set_xlabel(r"$\phi/\pi$")
    axes[1].set_ylabel(r"$D(p^{\mathrm{MC}},p^{\mathrm{analytic}})$")
    axes[1].set_title("(B) Trace-distance error across phase")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    fig.suptitle("NHS-CNG-PHASE-v2: qutrit clock phase benchmark", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nhs_cng_phase_clock_v2.pdf"), dpi=170)
    fig.savefig(os.path.join(fig_dir, "nhs_cng_phase_clock_v2.png"), dpi=170)
    plt.close(fig)


def plot_cng(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cng_nhs_cng_1_v2.csv"))
    betas = _sort_unique(arr["beta"])
    couplings = _sort_unique(arr["coupling"])
    beta_ref = _closest(betas, np.median(betas))
    coupling_ref = _closest(couplings, np.median(couplings))

    # representative row for population overlay
    rep_mask = np.isclose(arr["beta"], beta_ref) & np.isclose(arr["coupling"], coupling_ref)
    rep = arr[rep_mask][0]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Panel A: level populations at representative parameter point.
    level_idx = np.arange(3)
    width = 0.18
    ax_a.bar(level_idx - 1.5 * width, [rep["p0_ed"], rep["p1_ed"], rep["p2_ed"]], width=width, label="ED")
    ax_a.bar(level_idx - 0.5 * width, [rep["p0_k2"], rep["p1_k2"], rep["p2_k2"]], width=width, label="K2")
    ax_a.bar(level_idx + 0.5 * width, [rep["p0_k24"], rep["p1_k24"], rep["p2_k24"]], width=width, label="K2+K4")
    ax_a.bar(level_idx + 1.5 * width, [rep["p0_k246"], rep["p1_k246"], rep["p2_k246"]], width=width, label="K2+K4+K6")
    ax_a.set_xticks(level_idx)
    ax_a.set_xticklabels(["0", "1", "2"])
    ax_a.set_xlabel("Qutrit level index")
    ax_a.set_ylabel("Population")
    ax_a.set_title(rf"(A) Populations at $\beta={beta_ref:.2f}$, $g={coupling_ref:.2f}$")
    ax_a.legend(fontsize=8)

    # Panel B: trace-distance vs coupling for low/mid/high beta slices.
    beta_sel = np.unique(np.array([betas[0], betas[len(betas) // 2], betas[-1]], dtype=float))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(beta_sel)))
    for color, beta in zip(colors, beta_sel):
        sub = arr[np.isclose(arr["beta"], beta)]
        sub = np.sort(sub, order="coupling")
        gvals = np.asarray(sub["coupling"], dtype=float)
        td2 = np.asarray(sub["td_k2"], dtype=float)
        td24 = np.asarray(sub["td_k24"], dtype=float)
        ax_b.plot(gvals, td2, "--", color=color, linewidth=1.8)
        ax_b.plot(gvals, td24, "o-", color=color, linewidth=2.0, markersize=4.3)
    method_handles = [
        plt.Line2D([0], [0], linestyle="--", color="black", label="K2"),
        plt.Line2D([0], [0], linestyle="-", marker="o", color="black", label="K2+K4"),
    ]
    beta_handles = [plt.Line2D([0], [0], linestyle="-", color=c, label=fr"$\beta={b:.2f}$") for c, b in zip(colors, beta_sel)]
    leg_b = ax_b.legend(handles=method_handles, loc="upper left", fontsize=8)
    ax_b.add_artist(leg_b)
    ax_b.legend(handles=beta_handles, loc="upper right", fontsize=8, title="Slice")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$g$")
    ax_b.set_ylabel(r"$D(\rho_S^{\mathrm{ED}},\rho_S^{\mathrm{trunc}})$")
    ax_b.set_title("(B) Truncation error vs coupling")
    ax_b.grid(alpha=0.22, linewidth=0.6)

    # Panel C: validity map log10(D_K2 / D_K2+K4).
    improve = np.zeros((len(betas), len(couplings)), dtype=float)
    for i, beta in enumerate(betas):
        for j, g in enumerate(couplings):
            sub = arr[np.isclose(arr["beta"], beta) & np.isclose(arr["coupling"], g)]
            if len(sub) == 0:
                improve[i, j] = np.nan
            else:
                improve[i, j] = float(np.log10(max(float(sub["improvement_24"][0]), 1e-12)))
    im = ax_c.imshow(
        improve,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        extent=[couplings[0], couplings[-1], betas[0], betas[-1]],
    )
    ax_c.contour(
        couplings,
        betas,
        improve,
        levels=[0.0],
        colors="k",
        linewidths=1.0,
        linestyles="--",
    )
    ax_c.set_xlabel(r"$g$")
    ax_c.set_ylabel(r"$\beta$")
    ax_c.set_title(r"(C) Validity map: $\log_{10}(D_{K2}/D_{K2+K4})$")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("positive: K2+K4 improves")

    # Panel D: rescaled cumulants, showing K4 magnitude after g-rescaling.
    beta_unique = np.sort(np.unique(arr["beta"]).astype(float))
    a2_scaled = np.asarray(arr["alpha2"], dtype=float) / np.maximum(np.asarray(arr["coupling"], dtype=float) ** 2, 1e-15)
    a4_scaled = np.asarray(arr["alpha4"], dtype=float) / np.maximum(np.asarray(arr["coupling"], dtype=float) ** 4, 1e-15)
    for g in couplings:
        sub = arr[np.isclose(arr["coupling"], g)]
        sub = np.sort(sub, order="beta")
        ax_d.plot(sub["beta"], np.asarray(sub["alpha2"], dtype=float) / (g**2), color="#1f77b4", alpha=0.26, linewidth=1.0)
        ax_d.plot(sub["beta"], np.asarray(sub["alpha4"], dtype=float) / (g**4), color="#d62728", alpha=0.26, linewidth=1.0)
    mean_a2 = []
    std_a2 = []
    mean_a4 = []
    std_a4 = []
    for beta in beta_unique:
        mask = np.isclose(arr["beta"], beta)
        vals2 = a2_scaled[mask]
        vals4 = a4_scaled[mask]
        mean_a2.append(float(np.mean(vals2)))
        std_a2.append(float(np.std(vals2, ddof=1) if vals2.size > 1 else 0.0))
        mean_a4.append(float(np.mean(vals4)))
        std_a4.append(float(np.std(vals4, ddof=1) if vals4.size > 1 else 0.0))
    mean_a2 = np.asarray(mean_a2, dtype=float)
    std_a2 = np.asarray(std_a2, dtype=float)
    mean_a4 = np.asarray(mean_a4, dtype=float)
    std_a4 = np.asarray(std_a4, dtype=float)
    ax_d.plot(beta_unique, mean_a2, color="#1f77b4", linewidth=2.4, label=r"$\alpha_2/g^2$ mean")
    ax_d.fill_between(beta_unique, mean_a2 - std_a2, mean_a2 + std_a2, color="#1f77b4", alpha=0.16)
    ax_d.plot(beta_unique, mean_a4, color="#d62728", linewidth=2.4, label=r"$\alpha_4/g^4$ mean")
    ax_d.fill_between(beta_unique, mean_a4 - std_a4, mean_a4 + std_a4, color="#d62728", alpha=0.16)
    ax_d.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax_d.set_xlabel(r"$\beta$")
    ax_d.set_ylabel("Rescaled cumulant coefficient")
    ax_d.set_title("(D) Rescaled cumulants (coupling collapse)")
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.22, linewidth=0.6)

    fig.suptitle("NHS-CNG-1-v2: commuting non-Gaussian finite-spin benchmark", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(fig_dir, "nhs_cng_1_v2.pdf"), dpi=170)
    fig.savefig(os.path.join(fig_dir, "nhs_cng_1_v2.png"), dpi=170)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot commuting-only nested HS v2 suite")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base, "..", "results_v2"))
    data_dir = os.path.join(out_dir, "data")
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    plot_synth_nonpsd(data_dir, fig_dir)
    plot_phase_clock(data_dir, fig_dir)
    plot_cg(data_dir, fig_dir)
    plot_cng(data_dir, fig_dir)
    print("nested_hs_v2 figures generated")


if __name__ == "__main__":
    main()
