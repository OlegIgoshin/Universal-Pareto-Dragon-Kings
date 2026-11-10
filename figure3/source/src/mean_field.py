from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


@dataclass
class MeanFieldResult:
    probability: np.ndarray
    p0: float
    mean: float
    critical_p0: float
    converged: bool


def eta_tanh(x: np.ndarray | float, c: float, m0: float) -> np.ndarray:
    return c * np.tanh(np.asarray(x, dtype=np.float64) / m0)


def critical_p0(c: float) -> float:
    return 2.0 * c / (0.5 + c)


def kappa(c: float) -> float:
    if c <= 0:
        raise ValueError("c must be positive")
    return (0.5 - c) / (2.0 * c)


def _eta_average(probability: np.ndarray, kernel_fft: np.ndarray, fft_len: int) -> np.ndarray:
    mmax = probability.size - 1
    convolution = irfft(rfft(probability, fft_len) * kernel_fft, fft_len)
    return convolution[mmax : 2 * mmax + 1]


def fixed_p0_solution(
    p0: float,
    c: float,
    m0: float,
    mmax: int,
    tolerance: float,
    max_iterations: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    m = np.arange(mmax + 1, dtype=float)
    if initial is None:
        ratio = max(1e-8, min(0.999, 1.0 - p0))
        p = p0 * ratio ** m
        p[0] = p0
        p[1:] *= (1.0 - p0) / max(p[1:].sum(), np.finfo(float).tiny)
    else:
        p = initial.copy()
        p[0] = p0
        p[1:] *= (1.0 - p0) / max(p[1:].sum(), np.finfo(float).tiny)

    differences = np.arange(-mmax, mmax + 1, dtype=float)
    kernel = eta_tanh(differences, c, m0)
    fft_len = next_fast_len(3 * mmax + 1)
    kernel_fft = rfft(kernel, fft_len)
    eta_m = eta_tanh(m, c, m0)
    damping = 0.20
    tiny = np.finfo(float).tiny

    for _ in range(max_iterations):
        average = _eta_average(p, kernel_fft, fft_len)
        gain = 0.5 * (1.0 - p0) + average - p0 * eta_m
        loss = 0.5 - average
        loss[0] = 0.0
        np.clip(gain, tiny, 1.0, out=gain)
        np.clip(loss, tiny, 1.0, out=loss)
        log_tail = np.cumsum(np.log(gain[:-1]) - np.log(loss[1:]))
        log_tail -= np.max(log_tail)
        weights = np.exp(log_tail)
        candidate = np.empty_like(p)
        candidate[0] = p0
        candidate[1:] = (1.0 - p0) * weights / weights.sum()
        updated = (1.0 - damping) * p + damping * candidate
        change = float(np.abs(updated - p).sum())
        p = updated
        if change < tolerance:
            return p, True
    return p, False


def solve_for_mean(
    target_mean: float,
    c: float,
    m0: float,
    mmax: int,
    tolerance: float,
    max_iterations: int,
) -> MeanFieldResult:
    pc = critical_p0(c)
    m = np.arange(mmax + 1, dtype=float)
    lower = pc + 1e-7
    upper = 1.0 - 1e-8

    critical, ok = fixed_p0_solution(lower, c, m0, mmax, tolerance, max_iterations)
    critical_mean = float(np.dot(m, critical))
    if target_mean >= critical_mean:
        raise ValueError(
            f"Requested mean {target_mean:g} exceeds the finite-grid fluid mean "
            f"{critical_mean:g}. Increase mean_field_mmax."
        )

    best = critical
    best_p0 = lower
    best_mean = critical_mean
    converged = ok
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        trial, ok = fixed_p0_solution(
            midpoint, c, m0, mmax, tolerance, max_iterations, initial=best
        )
        trial_mean = float(np.dot(m, trial))
        if abs(trial_mean - target_mean) < max(1e-7, 1e-6 * max(target_mean, 1.0)):
            return MeanFieldResult(trial, midpoint, trial_mean, pc, ok)
        if trial_mean > target_mean:
            lower = midpoint
        else:
            upper = midpoint
        best, best_p0, best_mean = trial, midpoint, trial_mean
        converged = converged and ok

    return MeanFieldResult(best, best_p0, best_mean, pc, converged)
