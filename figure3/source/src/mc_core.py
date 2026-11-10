from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def advance_tanh(
    wealth: np.ndarray,
    order: np.ndarray,
    sweeps: int,
    c: float,
    m0: float,
    seed: int,
) -> None:
    """Random matching dynamics: each sweep pairs every agent once at random."""
    np.random.seed(seed)
    n = wealth.size
    for _ in range(sweeps):
        np.random.shuffle(order)
        for k in range(0, n - 1, 2):
            i = order[k]
            j = order[k + 1]
            difference = wealth[i] - wealth[j]
            eta = c * math.tanh(difference / m0)
            if np.random.random() < 0.5 + eta:
                if wealth[j] > 0:
                    wealth[i] += 1
                    wealth[j] -= 1
            elif wealth[i] > 0:
                wealth[i] -= 1
                wealth[j] += 1


def impose_total(wealth: np.ndarray, total: int, rng: np.random.Generator) -> None:
    difference = int(total - wealth.sum())
    if difference > 0:
        wealth += rng.multinomial(difference, np.full(wealth.size, 1.0 / wealth.size))
    elif difference < 0:
        remove = -difference
        while remove:
            nonempty = np.flatnonzero(wealth)
            count = min(remove, nonempty.size)
            selected = rng.choice(nonempty, count, replace=False)
            wealth[selected] -= 1
            remove -= count
    if int(wealth.sum()) != total:
        raise RuntimeError("Initialization failed to impose exact total wealth")


def stratified_sample_probability(
    probability: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    probability = probability / probability.sum()
    cdf = np.cumsum(probability)
    cdf[-1] = 1.0
    positions = (np.arange(n) + rng.random(n)) / n
    return np.searchsorted(cdf, positions).astype(np.int64)


def initialize_from_mean_field(
    probability: np.ndarray,
    n_agents: int,
    total_wealth: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wealth = stratified_sample_probability(probability, n_agents, rng)
    impose_total(wealth, total_wealth, rng)
    rng.shuffle(wealth)
    return wealth


def p0(wealth: np.ndarray) -> float:
    return float(np.mean(wealth == 0))


def distribution(wealth: np.ndarray, max_wealth: int | None = None) -> np.ndarray:
    hist = np.bincount(wealth)
    if max_wealth is not None:
        if hist.size <= max_wealth + 1:
            hist = np.pad(hist, (0, max_wealth + 1 - hist.size))
        else:
            overflow = hist[max_wealth + 1 :].sum()
            hist = hist[: max_wealth + 1]
            hist[-1] += overflow
    return hist.astype(float) / wealth.size


def save_state(path: Path, wealth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, wealth)
