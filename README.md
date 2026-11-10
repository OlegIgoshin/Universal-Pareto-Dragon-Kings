# Wealth-exchange Pareto scaling: Figures 2–5

Source-only reproduction package for Figures 2–5 and the **numerical checks reported in the Supporting Information** of *Universal Pareto law and wealth condensation in nonequilibrium wealth-exchange games*.

The repository contains Python source and simulation configurations, **not precomputed figure data, simulated trajectories, checkpoint files or the UN workbook**. Each invocation simulates/solves the stated equations and generates the selected figure anew. The high-wealth Figure 4F configurations reproduce the original long-run seeds and initialization choices. Stochastic trajectories need not reproduce the manuscript artwork pixel for pixel.

## Setup

Python 3.10 or later (use an environment supporting NumPy, SciPy and Numba):

```bash
python -m pip install -r requirements.txt
```

Run the commands with the same Python interpreter where dependencies were installed. `openpyxl` is required for Figure 5's UN Excel workbook.

## Run

```bash
# Quick functionality check; NOT publication-quality statistics
python run_all.py --profile smoke --jobs 2

# Full figures and numerical tests; expensive, especially Figure 4E/F
python run_all.py --profile publication --jobs 8

# Only Figures 4 and 5
python run_all.py --profile publication --figures 4 5 --jobs 8

# Figure 5, using official downloaded local data, with configurable city count K
python run_all.py --profile publication --figures 5 --city-K 30 \
  --city-workbook WUP2025-F21-DEGURBA-Cities_Pop.xlsx --out results_K30
```

Figure 5 defaults to `--city-K 40`. The parameter changes **both** country eligibility (at least K urban centres with population ≥50,000) and the number of largest ranked centres used within each country. For SI robustness checks run K=20, 30, 40 in separate output directories. The official input is the UN World Urbanization Prospects 2025, File 21 (DEGURBA). If `--city-workbook` is omitted, the program downloads it on first use into an ignored `.cache/figure5/` directory. Internet access is only needed to retrieve that observational input.

The `smoke` profile reduces Monte Carlo sampling and the Figure 5 null-simulation replicate count (1000 rather than one million); **do not compare smoke-mode p-values or convergence figures with the publication values**.

## Minimal outputs (default)

`run_all.py` generates all necessary intermediate data **during computation**, and after successful plotting retains only:

| Figure | Retained files inside `results/` | Purpose |
|---|---|---|
| 2 | `figure2/figure2_new_mc.pdf`, `new_mc_mean{4,10}.csv`, `new_mc_settings.json` | Plot, MC/theory values and settings |
| 3 | `figure3/Figure3_fresh.pdf`, `mc/panelA_summary.csv`, `C_diagnostics.json` | Plot, 3A error bars and 3C two-domain convergence (Table S5) |
| 4 | `figure4/assembly/figures/Figure4_replot.pdf`, `4D/D_summary.csv`, `4EF/EF_summary.csv`, panel `verification.json` files, and `4EF/Figure4F_*` | Plot, panel-level conservation checks, and Fig. 4F stationary wealth partition / high-wealth convergence |
| 5 | `figure5/figure5.pdf`, `country_statistics.csv`, `figure5_statistics.json` | Plot, country-level largest-city tests, exponents, KS screening and simulation-based p-values |
| All | `run_manifest.json` | Profile, selected figures and parameter K |

Intermediate full trajectories, checkpoint files, per-replicate block histories, intermediate plot CSVs, and extra image formats are removed **only after the figure is generated successfully**. For investigation, retaining full raw outputs, or replots without rerunning simulations, add `--keep-intermediates`. Temporary files can be large, especially in publication mode. Python bytecode, caches, and results are ignored by Git.

## Method and validation coverage

- **Fig. 2:** Poor-biased exchange, simulation distributions and continuum expressions from SI S1.
- **Fig. 3:** Mean-field and MC for 3A, matched near-critical continuum for 3B, and four bias-function comparisons with domain/flux convergence for 3C (SI S3, S5, Tables S2, S4–S5). Figure 3B uses prescribed mean wealths 1.4, 2.6, 3.1; the resulting δ values are computed, not fixed to 0.03, 0.003, 0.001.
- **Fig. 4:** Emergence, precondensed-state relaxation, four displayed means in 4D (1.38, 2.12, 2.97, 4.03), 13-point 4E/F ensembles, and 4F checks of initialization, production halves, and the fraction of extra wealth captured by the richest agent (SI S4–S5, Tables S1–S3). Figure 4D computes additional intermediate means but only the four SI-selected curves are plotted.
- **Fig. 5:** Pooled exponent, aggregate rank profile, F-statistic largest-city test, background KS screening, binomial exceedance test, Wilson intervals and the stated Monte Carlo tests (SI S6). Run different `--city-K` values for the expanded-country robustness analysis.

The source generates scientific plots. Final manuscript figure artwork may have received non-numerical typography or layout changes. See `figure4/README.md` for its exact high-wealth parameters and seeds.

## Repository layout

`run_all.py` is the single orchestrator; `figure2/`, `figure3/`, `figure4/`, and `figure5/` hold independent source; `cleanup_outputs.py` defines the default publication-oriented output manifest; `requirements.txt` declares dependencies.

## Reproducibility notes

Random seeds and Monte Carlo schedules are recorded in source and JSON configurations. The publication profile can require substantial CPU time. Full production agreement with the final artwork should be validated on the target environment. The dataset is third-party observational input and is intentionally not redistributed here.
