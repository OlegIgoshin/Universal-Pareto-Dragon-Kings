from __future__ import annotations

import numpy as np
import pandas as pd


def log_bin_probability(
    m: np.ndarray,
    p: np.ndarray,
    total_agent_samples: int,
    mmin: int = 1,
    mmax: int | None = None,
    bins_per_decade: int = 10,
    minimum_expected_counts: int = 100,
) -> pd.DataFrame:
    m = np.asarray(m, dtype=float)
    p = np.asarray(p, dtype=float)
    if mmax is None:
        mmax = int(np.nanmax(m))
    mask = (m >= mmin) & (m <= mmax) & np.isfinite(p) & (p >= 0)
    m = m[mask]
    p = p[mask]
    decades = np.log10(mmax / mmin)
    n_bins = max(8, int(np.ceil(decades * bins_per_decade)))
    edges = np.unique(np.floor(np.geomspace(mmin, mmax + 1, n_bins + 1)).astype(int))
    edges[0] = mmin
    if edges[-1] < mmax + 1:
        edges = np.r_[edges, mmax + 1]

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        in_bin = (m >= lo) & (m < hi)
        mass = float(p[in_bin].sum())
        if mass <= 0:
            continue
        expected = mass * total_agent_samples
        if expected < minimum_expected_counts:
            continue
        width = hi - lo
        mv = m[in_bin]
        weights = p[in_bin]
        representative = float(np.exp(np.sum(weights * np.log(mv)) / weights.sum()))
        rows.append(
            {
                "m_rep": representative,
                "p_bin": mass / width,
                "mass": mass,
                "lo": lo,
                "hi": hi,
                "expected_counts": expected,
            }
        )
    return pd.DataFrame(rows)


def summarize_panel_a(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mean_wealth, group in replicates.groupby("mean_wealth_fixed", sort=True):
        values = group["mean_p0_minus_pc"].to_numpy(float)
        rows.append(
            {
                "mean_wealth_fixed": mean_wealth,
                "mean_p0_minus_pc": float(values.mean()),
                "se_p0": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan,
                "sd_between_replicates": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "n_replicates": int(values.size),
                "mean_abs_half_drift": float(group["abs_half_drift"].mean()),
                "max_abs_half_drift": float(group["abs_half_drift"].max()),
                "mf_p0_minus_pc": float(group["mf_p0_minus_pc"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
