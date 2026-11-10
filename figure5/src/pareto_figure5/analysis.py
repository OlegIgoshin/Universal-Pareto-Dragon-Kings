"""Calculate the four panels from ranked populations."""

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np

from .data import City

DEFAULT_RANK_LIMIT = 40
FIT_THRESHOLD = 0.10


@dataclass(frozen=True)
class CountryResult:
    iso3: str
    country: str
    qualifying_cities: int
    cities: tuple[City, ...]
    normalized: np.ndarray
    background_sum: float
    background_ks: float
    background_p: float
    passes_background: bool
    dk_r: float
    dk_p: float
    reference_population: float


@dataclass(frozen=True)
class GroupResult:
    n: int
    above_median: int
    p_value: float
    wilson_low: float
    wilson_high: float


@dataclass(frozen=True)
class Figure5Analysis:
    countries: tuple[CountryResult, ...]
    ranks: np.ndarray
    density_edges: np.ndarray
    density_centers: np.ndarray
    density: np.ndarray
    pooled_q: float
    rank_geomean: np.ndarray
    rank_q25: np.ndarray
    rank_q75: np.ndarray
    rank_slope: float
    rank_amplitude: float
    all_countries: GroupResult
    screened_countries: GroupResult


def _ks_cdf(n: int, distance: float) -> float:
    """Finite-sample two-sided KS CDF (Durbin's matrix formula)."""
    if distance >= 1:
        return 1.0
    scaled = n * distance
    if scaled <= 0.5:
        return 0.0

    k = math.ceil(scaled)
    h = k - scaled
    size = 2 * k - 1
    matrix = np.zeros((size, size))
    inverse_factorial = np.array([1 / math.factorial(i) for i in range(size + 1)])
    for row in range(size):
        for col in range(min(size, row + 2)):
            matrix[row, col] = inverse_factorial[row - col + 1]
    powers = np.arange(1, size + 1)
    correction = h**powers * inverse_factorial[1:]
    matrix[:, 0] -= correction
    matrix[-1, :] -= correction[::-1]
    matrix[-1, 0] += max(2 * h - 1, 0)**size * inverse_factorial[-1]
    cdf = np.linalg.matrix_power(matrix, n)[k - 1, k - 1]
    cdf *= math.prod(i / n for i in range(1, n + 1))
    return float(np.clip(cdf, 0, 1))


def weighted_log_gaps(populations: np.ndarray) -> np.ndarray:
    """Rank-weighted log gaps along the last axis, in descending rank order."""
    return -np.arange(1, populations.shape[-1]) * np.diff(np.log(populations), axis=-1)


def _country_result(iso3: str, cities: list[City], ranks: np.ndarray) -> CountryResult:
    top = tuple(cities[:len(ranks)])
    populations = np.array([city.population for city in top], dtype=float)
    spacings = weighted_log_gaps(populations)
    background = spacings[1:]
    total = float(background.sum())
    if total <= 0:
        raise ValueError(f"{iso3}: background ranks have no population variation.")
    n_gaps = len(background)
    # For K ranked cities, the K-3 cumulative shares of K-2 background
    # gaps are uniform order statistics under independent Pareto sampling.
    cumulative = np.cumsum(background / total)[:-1]
    i = np.arange(len(cumulative))
    n = len(cumulative)
    distance = float(max(np.max((i + 1) / n - cumulative), np.max(cumulative - i / n)))
    fit_p = 1 - _ks_cdf(n, distance)
    dk_r = float(spacings[0] / (total / n_gaps))
    dk_p = math.exp(-n_gaps * math.log1p(dk_r / n_gaps))
    median = n_gaps * math.expm1(math.log(2) / n_gaps)
    reference_population = float(populations[1] * math.exp(median * total / n_gaps))
    return CountryResult(
        iso3=iso3,
        country=top[0].country,
        qualifying_cities=len(cities),
        cities=top,
        normalized=populations / populations[-1],
        background_sum=total,
        background_ks=distance,
        background_p=fit_p,
        passes_background=fit_p >= FIT_THRESHOLD,
        dk_r=dk_r,
        dk_p=dk_p,
        reference_population=reference_population,
    )


def _pooled_density(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For each country, ranks 2–39 are events. Rank 1 is right-censored
    # at rank 2, and rank 40 is the threshold. Group tied event times so
    # censoring at a tie removes observations only after the events.
    observed = profiles[:, 1:-1].ravel()
    censored = profiles[:, 1]
    times, inverse = np.unique(np.concatenate((observed, censored)), return_inverse=True)
    deaths = np.bincount(inverse[:observed.size], minlength=times.size)
    removals = np.bincount(inverse)
    at_risk = inverse.size - np.r_[0, np.cumsum(removals)[:-1]]
    hazard = deaths / at_risk
    survival_before = np.r_[1, np.cumprod(1 - hazard)[:-1]]
    mass = survival_before * hazard
    edges = np.geomspace(1, times[-1] * 1.001, 16)
    binned, _ = np.histogram(times, bins=edges, weights=mass)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers, binned / np.diff(edges)


def binomial_tail(above_median: int, n: int) -> float:
    """Probability of at least this many successes in n fair coin tosses."""
    return sum(math.comb(n, j) for j in range(above_median, n + 1)) / 2**n


def _group_result(countries: tuple[CountryResult, ...]) -> GroupResult:
    n = len(countries)
    if n == 0:
        raise ValueError("No eligible countries passed the background power-law check.")
    above_median = sum(country.dk_p < 0.5 for country in countries)
    p_value = binomial_tail(above_median, n)
    fraction = above_median / n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / n
    center = (fraction + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(fraction * (1 - fraction) / n + z**2 / (4 * n**2)) / denominator
    return GroupResult(
        n, above_median, p_value, max(0.0, center - half_width), min(1.0, center + half_width)
    )


def analyze(cities_by_country: dict[str, list[City]], rank_limit: int = DEFAULT_RANK_LIMIT) -> Figure5Analysis:
    """Analyze top K cities from every country with at least K qualifying cities."""
    if rank_limit < 4:
        raise ValueError("--rank-limit must be at least 4 (two or more background spacings).")
    ranks = np.arange(1, rank_limit + 1)
    countries = tuple(
        _country_result(iso3, cities, ranks)
        for iso3, cities in sorted(cities_by_country.items())
        if len(cities) >= rank_limit
    )
    if not countries:
        raise ValueError(f"The workbook contains no countries with {rank_limit} qualifying cities.")

    profiles = np.stack([country.normalized for country in countries])
    log_mean = np.log(profiles).mean(axis=0)
    q25, q75 = np.percentile(profiles, [25, 75], axis=0)
    slope, intercept = np.polyfit(np.log(ranks[1:]), log_mean[1:], 1)
    edges, centers, density = _pooled_density(profiles)
    pooled_q = (rank_limit - 2) * len(countries) / sum(country.background_sum for country in countries)
    screened = tuple(country for country in countries if country.passes_background)
    return Figure5Analysis(
        countries=countries,
        ranks=ranks,
        density_edges=edges,
        density_centers=centers,
        density=density,
        pooled_q=pooled_q,
        rank_geomean=np.exp(log_mean),
        rank_q25=q25,
        rank_q75=q75,
        rank_slope=float(slope),
        rank_amplitude=math.exp(intercept),
        all_countries=_group_result(countries),
        screened_countries=_group_result(screened),
    )
