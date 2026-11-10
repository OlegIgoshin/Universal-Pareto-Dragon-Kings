"""Seeded checks for the city-size analysis in the manuscript and SI."""

from __future__ import annotations

from math import exp, lgamma, log
import os

import numpy as np

from .analysis import FIT_THRESHOLD, Figure5Analysis, binomial_tail, weighted_log_gaps


REPLICATES = int(os.environ.get("PARETO_FIG5_REPLICATES", "1000000"))
BATCH_SIZE = 10_000


def _zipf_rank_one(observed: float, n_countries: int, last_rank: int) -> dict:
    seed = 2026110503
    rng = np.random.default_rng(seed)
    log_means = np.empty(REPLICATES)
    for start in range(0, REPLICATES, BATCH_SIZE):
        batch = log_means[start:start + BATCH_SIZE]
        uniform = rng.random((len(batch), n_countries))
        # Invert the CDF (1 - 1/x)**(last_rank - 1) and take the
        # geometric mean of the simulated maxima.
        with np.errstate(divide="ignore"):
            log_maxima = -np.log(-np.expm1(np.log(uniform) / (last_rank - 1)))
        batch[:] = log_maxima.mean(axis=1)
    exceedances = int(np.count_nonzero(log_means >= log(observed)))
    return {
        "seed": seed,
        "replicates": REPLICATES,
        "observed_geometric_mean": observed,
        "simulated_median": float(np.median(np.exp(log_means))),
        "exceedances": exceedances,
        "fraction_at_least_observed": exceedances / REPLICATES,
    }


def _common_exponent(background_sums: np.ndarray, n_background: int) -> dict:
    seed = 2026103102
    rng = np.random.default_rng(seed)
    n = len(background_sums)
    observed = 2 * n_background * (
        n * np.log(background_sums.mean()) - np.log(background_sums).sum()
    )
    exceedances = 0
    for start in range(0, REPLICATES, BATCH_SIZE):
        size = min(BATCH_SIZE, REPLICATES - start)
        # The unknown common Pareto rate cancels in the likelihood ratio.
        totals = rng.gamma(n_background, size=(size, n))
        null = 2 * n_background * (
            n * np.log(totals.mean(axis=1)) - np.log(totals).sum(axis=1)
        )
        exceedances += int(np.count_nonzero(null >= observed))
    return {
        "seed": seed,
        "replicates": REPLICATES,
        "likelihood_ratio": float(observed),
        "exceedances": exceedances,
        "p_value": (exceedances + 1) / (REPLICATES + 1),
    }


def _dk_calibration(dk_r: np.ndarray, dk_p: np.ndarray, last_rank: int) -> dict:
    seed = 2026110201
    rng = np.random.default_rng(seed)
    null = np.empty(REPLICATES)
    for start in range(0, REPLICATES, BATCH_SIZE):
        batch = null[start:start + BATCH_SIZE]
        draws = 1 + rng.pareto(1, size=(len(batch), last_rank))
        gaps = weighted_log_gaps(np.sort(draws, axis=1)[:, ::-1])
        batch[:] = gaps[:, 0] / gaps[:, 1:].mean(axis=1)

    null.sort()
    empirical = np.searchsorted(null, dk_r, side="left") / REPLICATES
    analytic = 1 - dk_p
    return {
        "seed": seed,
        "replicates": REPLICATES,
        "simulated_median": float(np.median(null)),
        "analytic_median": float((last_rank - 2) * np.expm1(log(2) / (last_rank - 2))),
        "max_absolute_difference_percentile_points": float(
            100 * np.max(np.abs(empirical - analytic))
        ),
        "above_median_classifications_match": bool(
            np.array_equal(empirical > 0.5, analytic > 0.5)
        ),
    }


def _fisher(log_p: np.ndarray) -> tuple[float, float]:
    half_statistic = -float(log_p.sum())
    if half_statistic == 0:
        return 0.0, 1.0
    # Chi-square survival for 2n degrees of freedom, evaluated in log space.
    terms = [j * log(half_statistic) - lgamma(j + 1) for j in range(len(log_p))]
    peak = max(terms)
    tail = exp(peak - half_statistic) * sum(exp(term - peak) for term in terms)
    return 2 * half_statistic, min(tail, 1.0)


def _upper_shift_ks(ordered: np.ndarray) -> np.ndarray:
    n = ordered.shape[-1]
    return np.max(ordered - np.arange(n) / n, axis=-1)


def _screen_calibration(
    observed: np.ndarray, exact_p: np.ndarray, passes: np.ndarray, last_rank: int
) -> dict:
    seed = 2026103101
    rng = np.random.default_rng(seed)
    n = last_rank - 3
    expected_upper_steps = np.arange(1, n + 1) / n
    null = np.empty(REPLICATES)
    for start in range(0, REPLICATES, BATCH_SIZE):
        batch = null[start:start + BATCH_SIZE]
        ordered = np.sort(rng.random((len(batch), n)), axis=1)
        # The cumulative normalized spacings have this uniform order
        # statistic distribution under the null.
        below = np.max(expected_upper_steps - ordered, axis=1)
        batch[:] = np.maximum(below, _upper_shift_ks(ordered))

    null.sort()
    exceedances = REPLICATES - np.searchsorted(null, observed, side="left")
    p = (exceedances + 1) / (REPLICATES + 1)
    simulated_passes = p >= FIT_THRESHOLD
    return {
        "seed": seed,
        "replicates": REPLICATES,
        "simulated_pass_count": int(np.count_nonzero(simulated_passes)),
        "screen_matches_exact_test": bool(np.array_equal(simulated_passes, passes)),
        "max_absolute_p_difference": float(np.max(np.abs(p - exact_p))),
    }


def _full_score_checks(dk_p: np.ndarray, passes: np.ndarray) -> dict:
    seed = 2026110202
    rng = np.random.default_rng(seed)
    groups = {"all": dk_p, "screened": dk_p[passes]}
    groups = {name: p for name, p in groups.items() if len(p)}
    observed = {
        name: float(_upper_shift_ks(np.sort(1 - p)))
        for name, p in groups.items()
    }
    exceedances = dict.fromkeys(groups, 0)
    for start in range(0, REPLICATES, BATCH_SIZE):
        size = min(BATCH_SIZE, REPLICATES - start)
        uniform = rng.random((size, len(dk_p)))
        for name, p in groups.items():
            ordered = np.sort(uniform[:, :len(p)], axis=1)
            null = _upper_shift_ks(ordered)
            exceedances[name] += int(np.count_nonzero(null >= observed[name]))

    results = {}
    for name, p in groups.items():
        fisher_statistic, fisher_p = _fisher(np.log(p))
        results[name] = {
            "n": len(p),
            "one_sided_ks_statistic": observed[name],
            "ks_exceedances": exceedances[name],
            "ks_p_value": (exceedances[name] + 1) / (REPLICATES + 1),
            "fisher_statistic": fisher_statistic,
            "fisher_p_value": fisher_p,
        }
    return {"seed": seed, "replicates": REPLICATES, "groups": results}


def _rank_sensitivity(populations: np.ndarray) -> list[dict]:
    n, last_rank = populations.shape
    gaps = weighted_log_gaps(populations)
    results = []
    for k in range(20, last_rank + 1):
        totals = gaps[:, 1:k - 1].sum(axis=1)
        log_p = -(k - 2) * np.log1p(gaps[:, 0] / totals)
        above = int(np.count_nonzero(log_p < -log(2)))
        _, fisher_p = _fisher(log_p)
        results.append({
            "last_rank": k,
            "n_countries": n,
            "above_median": above,
            "binomial_p_value": binomial_tail(above, n),
            "fisher_p_value": fisher_p,
        })
    return results


def supporting_checks(result: Figure5Analysis) -> dict[str, object]:
    """Recompute the additional checks from the same eligible countries."""
    countries = result.countries
    populations = np.array([[city.population for city in c.cities] for c in countries])
    last_rank = populations.shape[1]
    background_sums = np.array([c.background_sum for c in countries])
    dk_r = np.array([c.dk_r for c in countries])
    dk_p = np.array([c.dk_p for c in countries])
    passes = np.array([c.passes_background for c in countries])
    background_ks = np.array([c.background_ks for c in countries])
    background_p = np.array([c.background_p for c in countries])
    return {
        "zipf_rank_one": _zipf_rank_one(float(result.rank_geomean[0]), len(countries), last_rank),
        "common_exponent": _common_exponent(background_sums, last_rank - 2),
        "dk_calibration": _dk_calibration(dk_r, dk_p, last_rank),
        "screen_calibration": _screen_calibration(background_ks, background_p, passes, last_rank),
        "full_score_checks": _full_score_checks(dk_p, passes),
    }
