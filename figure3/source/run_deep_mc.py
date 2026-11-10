#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import log_bin_probability, summarize_panel_a
from mc_core import advance_tanh, distribution, initialize_from_mean_field, p0, save_state
from mean_field import solve_for_mean


def load_config(profile: str) -> dict:
    path = ROOT / "configs" / f"{profile}.json"
    if not path.exists():
        path = Path(profile)
    with path.open() as handle:
        return json.load(handle)


def solve_initial(mean_wealth: float, cfg: dict):
    model = cfg["model"]
    result = solve_for_mean(
        mean_wealth,
        model["eta_infinity"],
        model["m0"],
        model["mean_field_mmax"],
        model["mean_field_tolerance"],
        model["mean_field_max_iterations"],
    )
    if not result.converged:
        print(f"WARNING: MF initialization for mean={mean_wealth:g} did not meet tolerance")
    return result


def run_panel_a(cfg: dict, outdir: Path) -> None:
    model = cfg["model"]
    panel = cfg["panel_A"]
    n = int(model["n_agents"])
    c = float(model["eta_infinity"])
    m0 = float(model["m0"])
    seed_base = int(model["seed_base"])
    reps = int(panel["replicates"])
    means = panel["means"]
    burn_list = panel["burn_sweeps"]
    sample_list = panel["sample_sweeps"]
    trace_every = int(panel["trace_every"])

    rep_rows = []
    trace_rows = []
    mf_rows = []
    for idx, target_mean in enumerate(means):
        mf = solve_initial(float(target_mean), cfg)
        mf_rows.append(
            {
                "target_mean": target_mean,
                "mf_mean": mf.mean,
                "mf_p0": mf.p0,
                "mf_p0_minus_pc": mf.p0 - mf.critical_p0,
                "pc": mf.critical_p0,
            }
        )
        burn = int(burn_list[idx])
        sample = int(sample_list[idx])
        total_wealth = int(round(target_mean * n))
        for rep in range(reps):
            seed = seed_base + 100000 * idx + rep
            wealth = initialize_from_mean_field(mf.probability, n, total_wealth, seed)
            order = np.arange(n, dtype=np.int64)
            t0 = time.time()
            advance_tanh(wealth, order, burn, c, m0, seed + 1)

            samples = []
            remaining = sample
            block_index = 0
            while remaining > 0:
                block = min(trace_every, remaining)
                advance_tanh(wealth, order, block, c, m0, seed + 1000 + block_index)
                current_p0 = p0(wealth)
                samples.append(current_p0)
                trace_rows.append(
                    {
                        "mean_wealth_fixed": total_wealth / n,
                        "replicate": rep,
                        "sample_sweep": sample - remaining + block,
                        "p0": current_p0,
                        "p0_minus_pc": current_p0 - mf.critical_p0,
                    }
                )
                remaining -= block
                block_index += 1

            values = np.asarray(samples, dtype=float)
            half = max(1, values.size // 2)
            first = float(values[:half].mean())
            second = float(values[half:].mean())
            rep_rows.append(
                {
                    "mean_wealth_fixed": total_wealth / n,
                    "replicate": rep,
                    "mean_p0": float(values.mean()),
                    "mean_p0_minus_pc": float(values.mean() - mf.critical_p0),
                    "mf_p0": mf.p0,
                    "mf_p0_minus_pc": mf.p0 - mf.critical_p0,
                    "first_half_p0": first,
                    "second_half_p0": second,
                    "half_drift": second - first,
                    "abs_half_drift": abs(second - first),
                    "burn_sweeps": burn,
                    "sample_sweeps": sample,
                    "trace_every": trace_every,
                    "runtime_sec": time.time() - t0,
                }
            )
            if panel.get("save_final_states", False):
                save_state(outdir / "states" / f"A_mean{target_mean:.4f}_rep{rep:02d}.npy", wealth)
            print(
                f"A mean={target_mean:.4f} rep={rep+1}/{reps}: "
                f"p0-pc={values.mean()-mf.critical_p0:.6g}, "
                f"half-drift={second-first:+.3g}"
            )

    rep_df = pd.DataFrame(rep_rows)
    trace_df = pd.DataFrame(trace_rows)
    mf_df = pd.DataFrame(mf_rows)
    rep_df.to_csv(outdir / "panelA_replicates.csv", index=False)
    trace_df.to_csv(outdir / "panelA_traces.csv", index=False)
    mf_df.to_csv(outdir / "panelA_mean_field_points.csv", index=False)
    summarize_panel_a(rep_df).to_csv(outdir / "panelA_summary.csv", index=False)


def run_panel_b(cfg: dict, outdir: Path) -> None:
    model = cfg["model"]
    panel = cfg["panel_B"]
    n = int(model["n_agents"])
    c = float(model["eta_infinity"])
    m0 = float(model["m0"])
    seed_base = int(model["seed_base"]) + 50000000
    reps = int(panel["replicates"])
    max_hist = int(panel["max_hist_wealth"])

    rep_rows = []
    mf_rows = []
    all_binned = []
    all_raw = []

    for idx, target_mean in enumerate(panel["means"]):
        mf = solve_initial(float(target_mean), cfg)
        mf_rows.append(
            {
                "target_mean": target_mean,
                "mf_mean": mf.mean,
                "mf_p0": mf.p0,
                "mf_p0_minus_pc": mf.p0 - mf.critical_p0,
            }
        )
        burn = int(panel["burn_sweeps"][idx])
        sample = int(panel["sample_sweeps"][idx])
        hist_every = int(panel["hist_every"])
        total_wealth = int(round(target_mean * n))
        aggregate = np.zeros(max_hist + 1, dtype=float)
        total_agent_samples = 0

        for rep in range(reps):
            seed = seed_base + 100000 * idx + rep
            wealth = initialize_from_mean_field(mf.probability, n, total_wealth, seed)
            order = np.arange(n, dtype=np.int64)
            t0 = time.time()
            advance_tanh(wealth, order, burn, c, m0, seed + 1)

            hist_sum = np.zeros(max_hist + 1, dtype=float)
            n_hist = 0
            n_steps = sample // hist_every
            p0_samples = []
            for block_index in range(n_steps):
                advance_tanh(wealth, order, hist_every, c, m0, seed + 1000 + block_index)
                dist = distribution(wealth, max_hist)
                hist_sum += dist
                p0_samples.append(float(dist[0]))
                n_hist += 1

            replicate_distribution = hist_sum / max(1, n_hist)
            aggregate += replicate_distribution
            total_agent_samples += n_hist * n
            p0_values = np.asarray(p0_samples)
            half = max(1, p0_values.size // 2)
            first = float(p0_values[:half].mean())
            second = float(p0_values[half:].mean())
            rep_rows.append(
                {
                    "mean_wealth_fixed": total_wealth / n,
                    "replicate": rep,
                    "mean_p0": float(p0_values.mean()),
                    "mean_p0_minus_pc": float(p0_values.mean() - mf.critical_p0),
                    "half_drift": second - first,
                    "abs_half_drift": abs(second - first),
                    "burn_sweeps": burn,
                    "sample_sweeps": sample,
                    "hist_every": hist_every,
                    "n_histograms": n_hist,
                    "runtime_sec": time.time() - t0,
                }
            )
            print(
                f"B mean={target_mean:.4f} rep={rep+1}/{reps}: "
                f"p0-pc={p0_values.mean()-mf.critical_p0:.6g}"
            )

        aggregate /= reps
        m = np.arange(max_hist + 1)
        raw = pd.DataFrame(
            {
                "mean_wealth": target_mean,
                "m": m,
                "p_mc": aggregate,
            }
        )
        all_raw.append(raw)
        binned = log_bin_probability(
            m,
            aggregate,
            total_agent_samples=total_agent_samples,
            mmin=1,
            mmax=max_hist,
            bins_per_decade=int(panel["bins_per_decade"]),
            minimum_expected_counts=int(panel["minimum_expected_counts_per_bin"]),
        )
        binned.insert(0, "mean_wealth", target_mean)
        all_binned.append(binned)

    pd.DataFrame(rep_rows).to_csv(outdir / "panelB_replicates.csv", index=False)
    pd.DataFrame(mf_rows).to_csv(outdir / "panelB_mean_field_points.csv", index=False)
    pd.concat(all_raw, ignore_index=True).to_csv(outdir / "panelB_raw_distributions.csv", index=False)
    pd.concat(all_binned, ignore_index=True).to_csv(outdir / "panelB_logbinned_distributions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep MC for Figure 1 panels A and B")
    parser.add_argument("--profile", default="publication", help="publication, smoke, or path to JSON config")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--panels", default="AB", choices=["A", "B", "AB"])
    args = parser.parse_args()

    cfg = load_config(args.profile)
    outdir = Path(args.out) if args.out else ROOT / "results" / Path(args.profile).stem
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_used.json").write_text(json.dumps(cfg, indent=2))

    if "A" in args.panels:
        run_panel_a(cfg, outdir)
    if "B" in args.panels:
        run_panel_b(cfg, outdir)


if __name__ == "__main__":
    main()
