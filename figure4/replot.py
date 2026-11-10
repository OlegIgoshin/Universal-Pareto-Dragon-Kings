#!/usr/bin/env python3
"""Publication plotter for revised 3-column six-panel dragon-king figure.

Layout:
    A | C | E
    B | D | F

A/B and C/D use the same rectangular geometry. E/F are square.
C and D share a logarithmic wealth-m axis from 10^0 to 10^3.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DERIVED = ROOT / "data"
OUT = ROOT / "plots"
PC = 2/3
KAPPA = 0.5
N = 300


def config():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        # 50% larger than the previous 6.2-pt figure legends.
        "legend.fontsize": 9.3,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_summary():
    p = DERIVED / "EF_summary.csv"
    if not p.exists():
        raise SystemExit("Missing newly generated EF_summary.csv. Run run_all.py first.")
    return pd.read_csv(p)


def style(ax, lab):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.text(-0.14, 1.045, lab, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")


def main():
    import argparse
    global DATA, DERIVED, OUT
    ap=argparse.ArgumentParser(description="Plot Figure 4 from numerical data freshly computed by run_all.py.")
    ap.add_argument('--data',type=Path,default=ROOT/'data')
    ap.add_argument('--outdir',type=Path,default=ROOT/'plots')
    args=ap.parse_args(); DATA=args.data; DERIVED=args.data; OUT=args.outdir
    config(); OUT.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(DATA / "selected_trajectories_early_logsample.csv")
    obs = pd.read_csv(DATA / "global_observables.csv")
    decomp = pd.read_csv(DATA / "stationary_distribution_dense.csv")
    tails = pd.read_csv(DATA / "finite_size_tail_curves.csv")
    mf = pd.read_csv(DATA / "critical_meanfield_solution.csv")
    cross = pd.read_csv(DATA / "critical_bulk_matching_revised.csv")
    summary = load_summary()
    mux = float(cross.loc[cross["N"] == N, "mbar_x_revised"].iloc[0])

    # Two rectangular columns plus a square E/F column. At this figure size the
    # first two columns reproduce approximately the A/B aspect ratio of the old figure.
    fig = plt.figure(figsize=(11.8, 5.75))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.65, 1.65, 1.0], height_ratios=[1, 1])
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.105, top=0.975,
                        wspace=0.34, hspace=0.39)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[0, 1]); axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[0, 2]); axF = fig.add_subplot(gs[1, 2])
    axE.set_box_aspect(1); axF.set_box_aspect(1)

    # A. Spontaneous emergence trajectories.
    x = traj["time_sweeps"].to_numpy(float) + 1
    for col in [c for c in traj.columns if c.startswith("ordinary")]:
        axA.loglog(x, traj[col].to_numpy(float) + 1, lw=0.65, alpha=0.32, color="0.55")
    l2, = axA.loglog(x, traj["final second-richest"].to_numpy(float) + 1,
                     lw=1.35, label="second-richest at final time")
    l1, = axA.loglog(x, traj["final richest"].to_numpy(float) + 1,
                     lw=1.55, label="richest at final time")
    axA.set(xlabel=r"time $t+1$ (sweeps)", ylabel=r"wealth $m_i+1$")
    axA.legend(handles=[l2, l1], frameon=False, loc="upper left", handlelength=1.8)

    # B. Empty fraction in same emergence run.
    xb = obs["time_sweeps"].to_numpy(float) + 1
    axB.semilogx(xb, obs["p0"].to_numpy(float), color="k", lw=1.05, label=r"$p_0(t)$")
    axB.axhline((1 - 1/N) * PC, color="0.35", ls="--", lw=1.0,
                label=r"$(1-1/N)p_c$")
    axB.set(xlabel=r"time $t+1$ (sweeps)", ylabel=r"empty-player fraction $p_0$", ylim=(0, 0.73))
    axB.legend(frameon=False, loc="lower right", handlelength=1.8)

    # C. Stationary distribution: use wealth m directly and omit m=0 on log x.
    m = decomp["m"].to_numpy(float)
    for col, label in [
        ("p_total_mean", "total"),
        ("p_liquid_population_mean", "bulk contribution"),
        ("p_dragon_population_mean", "outlier contribution"),
    ]:
        y = decomp[col].to_numpy(float)
        mask = (m >= 1) & (y > 0)
        axC.loglog(m[mask], y[mask], label=label)
    refm = np.geomspace(1, 1000, 300)
    axC.loglog(refm, KAPPA / refm**2, "k--", lw=1.0, label=r"$\kappa/m^2$")
    axC.set(xlabel=r"wealth $m$", ylabel=r"$p(m)$",
            xlim=(1, 1000), ylim=(1e-8, 2e-1))
    axC.legend(frameon=False, loc="lower left", ncol=2,
               columnspacing=0.75, handlelength=1.55, borderaxespad=0.2)

    # D. Compensated cumulative tail, exactly same wealth-m x axis as C.
    # Publication panel D: plot only the four means listed in SI Table S1.
    # Other simulated means remain available in the source CSV for diagnostics.
    for target_mu in (1.38, 2.12, 2.97, 4.03):
        d = tails[np.isclose(tails["mean_wealth"], target_mu, atol=0.005)]
        if d.empty:
            raise ValueError(f"Missing publication Figure 4D mean wealth {target_mu}")
        mu = float(d["mean_wealth"].iloc[0])
        mask = (d["m"] >= 1) & (d["S"] > 0) & (d["m"] <= 1000)
        axD.semilogx(d.loc[mask, "m"], d.loc[mask, "m"] * d.loc[mask, "S"] / KAPPA,
                     label=fr"$\bar m={mu:.2f}$")
    mfmask = (mf["m"] >= 1) & (mf["m"] <= 1000)
    axD.semilogx(mf.loc[mfmask, "m"],
                 mf.loc[mfmask, "m"] * mf.loc[mfmask, "S_meanfield"] / KAPPA,
                 "k-.", lw=1.1, label=r"critical MF $S_c$")
    axD.axhline(1, ls="--", color="0.35", lw=1.0, label=r"$S_c(m)\simeq\kappa/m$")
    axD.set(xlabel=r"wealth $m$", ylabel=r"$mS_N(m)/\kappa$",
            xlim=(1, 1000), ylim=(0, 3.0))
    axD.legend(frameon=False, loc="upper left", ncol=2,
               columnspacing=0.65, handlelength=1.55, borderaxespad=0.2)

    # E. Recomputed stationary total empty fraction.
    axE.errorbar(summary["mean_wealth"], summary["p0_mean"], yerr=summary["p0_se"],
                 fmt="o-", capsize=2.2, lw=1.35, ms=4.4)
    axE.axhline(PC, ls="--", color="0.25", lw=1.0, label=r"$p_c$")
    axE.axhline((1 - 1/N) * PC, ls=":", color="0.45", lw=1.2,
                label=r"$(1-1/N)p_c$")
    axE.axvline(mux, ls=":", color="0.15", lw=1.0, label=r"$\kappa\ln(\kappa N)$")
    axE.set(xlabel=r"mean wealth $\bar m=M/N$", ylabel=r"empty-player fraction $p(0)$",
            xlim=(1.25, 4.15), ylim=(0.6638, 0.6832))
    axE.legend(frameon=False, loc="upper right", handlelength=1.55)

    # F. Per-agent wealth split, computed from freshly sampled maximum wealth.
    # max_fraction_mean is E[max wealth / M] from each trajectory, M=N*mu.
    # Therefore E[max wealth]/N = mu*max_fraction_mean.
    # The remaining bulk contribution is mu - E[max wealth]/N.
    x = summary["mean_wealth"].to_numpy(float)
    frac = summary["max_fraction_mean"].to_numpy(float)
    frac_se = summary["max_fraction_se"].to_numpy(float)
    outlier = x * frac
    bulk = x - outlier
    yerr = x * frac_se
    axF.errorbar(x, outlier, yerr=yerr, fmt="o-", capsize=2.2,
                 lw=1.35, ms=4.4, label=r"$\langle m_{\max}\rangle/N$")
    axF.errorbar(x, bulk, yerr=yerr, fmt="s-", capsize=2.2,
                 lw=1.35, ms=4.0, label=r"$(M-\langle m_{\max}\rangle)/N$")
    axF.axvline(mux, ls=":", color="0.15", lw=1.0)
    # Historical Figure4F used a translated unit-slope line as a visual guide,
    # not an absolute theoretical prediction for the condensate wealth.
    if np.any(np.isclose(x, 12.0)):
        bulk12 = float(bulk[np.argmin(abs(x - 12.0))])
        guide_x = np.array([6.0, 12.0])
        axF.plot(guide_x, guide_x - bulk12 + 0.5, "--", color="0.4",
                 lw=1.0, label="unit-slope guide")
    axF.set(xlabel=r"mean wealth $\bar m=M/N$", ylabel=r"wealth per agent",
            xlim=(0, 12.6), ylim=(0, 11.0))
    axF.legend(frameon=False, loc="upper left", fontsize=7.2)

    for ax, lab in [(axA,"A"),(axB,"B"),(axC,"C"),(axD,"D"),(axE,"E"),(axF,"F")]:
        style(ax, lab)

    stem = OUT / "Figure4_replot"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix('.pdf'))


if __name__ == "__main__":
    main()
