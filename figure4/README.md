# Figure 4 standalone source

This folder contains the Monte Carlo simulations and plotting code for
Figure 4A--F. No precomputed simulation data are needed.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run all panels (reduced check):

```bash
python run_figure4.py --profile smoke --jobs 2
```

Run publication configuration:

```bash
python run_figure4.py --profile publication --jobs 8
```

Panels A--B now record $t=0,1,2,4,8,...$ with a continuous RNG stream.
Figure 4C creates its own initial backgrounds. D and E--F are computed
from scratch. The publication configuration can be CPU-intensive.

## Figure 4F high-wealth historical configuration

The three high-wealth points, $M/N=6,8,12$, now use the **original**
`Figure4F_wealth_partition.zip` simulation configurations:

- $N=300$, step bias $\eta_\infty=1/4$, 12 independent runs per point.
- Six broad (`exponential`) and six precondensed starts per mean. The
  latter initially place 725 wealth units in the background; the remaining
  wealth is concentrated in one agent. This is an initialization choice,
  **not** a predicted bulk plateau.
- Historical seeds: `2026110200+r`, `2026120200+r`, `2026130200+r`
  respectively, for `r=0,...,11`.
- 8,000,000 burn-in sweeps and 3,000,000 production sweeps per run,
  with 20,000-sweep blocks and histograms sampled every 500 sweeps.
- The 10 lower-wealth points retain their existing MC schedules and seeds.

The program computes both $\langle m_{\max}\rangle/N$ and
$(M-\langle m_{\max}\rangle)/N$ and their standard errors across
independent trajectories. Figure 4F's dashed unit-slope line is a visual
slope guide with an arbitrary intercept, not a parameter-free theory curve.
The vertical crossover guide is $\kappa\ln(\kappa N)$.

After a **complete publication** run, `4EF/` also contains:

- `Figure4F_highwealth_partition.csv` — 6, 8, 12 partition means and SEMs;
- `Figure4F_wealth_capture.json` — fraction of added wealth (6 to 12)
  going to the richest agent and propagated SEM;
- `Figure4F_convergence_checks.csv` — broad/precondensed initial-state
  comparisons and first/second production-half checks.

These are newly computed outputs, **not** input data. Historical numerical
values can differ slightly across environments; full production-scale
numerical agreement has not been independently verified here.

## Minimal top-level results

When invoked using the repository's top-level `run_all.py`, Figure 4 retains
only the final PDF, aggregate D / E–F tables, panel verification reports,
and the high-wealth partition / convergence tests reported in SI. Raw
checkpoints and block metrics exist throughout simulation and plotting, but
are removed after successful completion. To preserve them, use
`python run_all.py ... --keep-intermediates`. Running this subdirectory's
`run_figure4.py` directly does not apply the top-level pruning policy.
