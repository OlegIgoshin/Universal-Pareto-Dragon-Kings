"""Draw the four manuscript panels from the computed statistics."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

from .analysis import Figure5Analysis


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
FIGURE_SIZE = 8.7 / 2.54  # PNAS single-column width, in inches
LEGEND_STYLE = {
    "frameon": False,
    "handlelength": 0.85,
    "handletextpad": 0.35,
    "labelspacing": 0.1,
    "borderaxespad": 0.25,
}


def _pooled_density(ax: Axes, result: Figure5Analysis) -> None:
    positive = result.density > 0
    ax.loglog(
        result.density_centers[positive], result.density[positive],
        "o", ms=3.2, color=BLUE, label="Pooled", zorder=3,
    )
    x = np.geomspace(1, result.density_edges[-1], 250)
    exponent = -(1 + result.pooled_q)
    ax.loglog(x, result.pooled_q * x**exponent, color=ORANGE, lw=0.9,
              label=rf"Fit: $x^{{{exponent:.2f}}}$")
    ax.loglog(x, x**-2, "k--", lw=0.8, label=r"$x^{-2}$")
    ax.set(xlim=(0.9, result.density_edges[-1] * 1.08),
           xlabel=rf"$x=m/m_{{{len(result.ranks)}}}$", ylabel=r"Density, $p(x)$")
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 50]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1]))
    ax.legend(loc="lower left", **LEGEND_STYLE)


def _rank_axes(ax: Axes, maximum: float, last_rank: int) -> None:
    ax.set(xlim=(0.83, 1.15 * last_rank), ylim=(0.85, maximum),
           xlabel=r"Rank, $k$", ylabel=rf"$x=m_k/m_{{{last_rank}}}$")
    ax.xaxis.set_major_locator(FixedLocator([1, 5, last_rank]))
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))


def _aggregate_rank(ax: Axes, result: Figure5Analysis) -> None:
    ranks = result.ranks[1:]
    ax.fill_between(ranks, result.rank_q25[1:], result.rank_q75[1:],
                    color=BLUE, alpha=0.14, linewidth=0)
    ax.loglog(ranks, result.rank_geomean[1:], "o-", color=BLUE, lw=1, ms=2.8,
              label="Mean", zorder=3)
    largest = result.rank_geomean[0]
    ax.scatter([1], [largest], s=17, color=RED, edgecolor="white", lw=0.3, zorder=4)
    ax.annotate("Largest city", (1, largest), xytext=(4, 2),
                textcoords="offset points", va="center")
    k = np.geomspace(1, result.ranks[-1], 250)
    ax.loglog(k, result.rank_amplitude * k**result.rank_slope,
              color=ORANGE, lw=0.9, label=rf"Fit: $k^{{{result.rank_slope:.2f}}}$")
    ax.loglog(k, result.ranks[-1] / k, "k--", lw=0.8, label=r"$k^{-1}$")
    _rank_axes(ax, 100, len(result.ranks))
    ax.legend(loc="lower left", **LEGEND_STYLE)


def _individual_ranks(ax: Axes, result: Figure5Analysis) -> None:
    ranks = result.ranks[1:]
    examples = {"FRA": ("France", BLUE), "THA": ("Thailand", ORANGE),
                "ETH": ("Ethiopia", GREEN), "BRA": ("Brazil", RED)}
    for country in result.countries:
        x = country.normalized
        ax.loglog(ranks, x[1:], color="#bdbdbd", lw=0.45, alpha=0.5, zorder=1)
        ax.scatter([1], [x[0]], s=9, color="#999999", alpha=0.6, zorder=2)

    profiles = {country.iso3: country.normalized for country in result.countries}
    for iso3, (label, color) in examples.items():
        x = profiles[iso3]
        ax.loglog(ranks, x[1:], color=color, lw=0.8, label=label, zorder=3)
        ax.scatter([1], [x[0]], s=17, color=color, edgecolor="white", lw=0.3, zorder=4)
    _rank_axes(ax, 600, len(result.ranks))
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.94), **LEGEND_STYLE)


def _aggregate_excess(ax: Axes, result: Figure5Analysis) -> None:
    groups = [result.all_countries, result.screened_countries]
    proportions = np.array([group.above_median / group.n for group in groups])
    lower = np.array([group.wilson_low for group in groups])
    upper = np.array([group.wilson_high for group in groups])
    ax.bar([0, 1], 100 * proportions, width=0.7, color=[BLUE, ORANGE], alpha=0.9)
    ax.errorbar([0, 1], 100 * proportions,
                yerr=100 * np.array([proportions - lower, upper - proportions]),
                fmt="none", color="black", capsize=2.4, lw=0.8)
    ax.axhline(50, color="black", ls="--", lw=0.8)
    for position, (group, proportion) in enumerate(zip(groups, proportions)):
        ax.text(position, 25, f"{group.above_median}/{group.n}\n{proportion:.0%}",
                color="white", ha="center", va="center", fontsize=9)
        ax.text(position, 100 * group.wilson_high + 3, f"{group.p_value:.4g}",
                ha="center", va="bottom")
    ax.set_xticks([0, 1], ["All countries", "Passed\npower-law"])
    ax.set(xlim=(-0.58, 1.58), ylim=(0, 100))
    ax.set_ylabel("Above median (%)", labelpad=2)
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))


def write_figure(result: Figure5Analysis, output_dir: Path) -> None:
    """Write vector PDF/SVG and a 600 dpi PNG of Figure 5."""
    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "pareto-figure5",
        "savefig.facecolor": "white",
    }):
        fig = Figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2).flat
        fig.subplots_adjust(left=0.145, right=0.99, bottom=0.112, top=0.985,
                            wspace=0.4, hspace=0.35)
        panels = [_pooled_density, _aggregate_rank, _individual_ranks, _aggregate_excess]
        for index, (ax, plot_panel) in enumerate(zip(axes, panels)):
            plot_panel(ax, result)
            ax.set_box_aspect(1)
            ax.tick_params(which="major", length=2.4, pad=2)
            ax.tick_params(which="minor", length=1.4)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            fig.text(0.015 + 0.5 * (index % 2), ax.get_position().y1 + 0.009,
                     "ABCD"[index], fontsize=10, va="top")

        fig.savefig(output_dir / "figure5.pdf", metadata={"CreationDate": None})
        fig.savefig(output_dir / "figure5.png", dpi=600)
        fig.savefig(output_dir / "figure5.svg", metadata={"Date": None})
