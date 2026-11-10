"""Publication-grade Monte Carlo runner for the unfair-exchange model.

This runner reuses the Numba update kernel from ``unfair_exchange.simulation``
but separates a fixed burn-in phase from a fixed production phase.  It pools
production histograms across many samples, records top order statistics and
participation ratios, checkpoints sparsely, and is designed for multi-million
sweep runs that may be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from mc_kernel import _advance_builtin, NUMBA_AVAILABLE


@dataclass
class Spec:
    run_id: str
    study: str
    N: int
    M: int
    seed: int = 12345
    c: float = 0.25
    bias_kind: str = "step"
    x0: float = 1.0
    initial: str = "exponential"
    initial_bulk_mu: float | None = None
    burnin_sweeps: int = 1_000_000
    production_sweeps: int = 1_000_000
    block_sweeps: int = 20_000
    sample_every: int = 500
    order_every: int = 1000
    top_k: int = 10
    checkpoint_every_blocks: int = 10
    output_directory: str = "publication_outputs/run"
    mu_c_reference: float | None = None
    mu_ratio: float | None = None
    notes: str = ""

    @property
    def mu(self) -> float:
        return self.M / self.N


def _load_spec(path: Path) -> Spec:
    obj = json.loads(path.read_text())
    return Spec(**obj)


def _fingerprint(spec: Spec) -> str:
    # Deliberately omit production_sweeps and checkpoint frequency so a run can
    # be safely extended after inspecting preliminary output.
    immutable = {
        "run_id": spec.run_id, "study": spec.study, "N": spec.N, "M": spec.M,
        "seed": spec.seed, "c": spec.c, "bias_kind": spec.bias_kind, "x0": spec.x0,
        "initial": spec.initial, "initial_bulk_mu": spec.initial_bulk_mu, "burnin_sweeps": spec.burnin_sweeps,
        "block_sweeps": spec.block_sweeps, "sample_every": spec.sample_every,
        "order_every": spec.order_every, "top_k": spec.top_k,
    }
    return hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()


def _exponential_exact(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Broad uncondensed initial state with exactly M units of wealth."""
    if M == 0:
        return np.zeros(N, dtype=np.int64)
    mu = M / N
    ratio = mu / (1.0 + mu)
    u = (np.arange(N) + rng.random(N)) / N
    w = np.floor(np.log1p(-u) / math.log(ratio)).astype(np.int64)
    diff = int(M - w.sum())
    if diff > 0:
        w += rng.multinomial(diff, np.full(N, 1.0 / N))
    elif diff < 0:
        remove = -diff
        while remove:
            nz = np.flatnonzero(w)
            take = min(remove, nz.size)
            selected = rng.choice(nz, take, replace=False)
            w[selected] -= 1
            remove -= take
    return w


def _initialize(spec: Spec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    if spec.initial == "random":
        w = rng.multinomial(spec.M, np.full(spec.N, 1.0 / spec.N)).astype(np.int64)
    elif spec.initial == "exponential":
        w = _exponential_exact(spec.N, spec.M, rng)
    elif spec.initial == "precondensed":
        if spec.initial_bulk_mu is None:
            if spec.mu_c_reference is None:
                raise ValueError("precondensed initialization requires initial_bulk_mu or mu_c_reference")
            bulk_mu = float(spec.mu_c_reference)
        else:
            bulk_mu = float(spec.initial_bulk_mu)
        bulk_M = min(spec.M, max(0, int(round(spec.N * bulk_mu))))
        w = _exponential_exact(spec.N, bulk_M, rng)
        excess = int(spec.M - bulk_M)
        if excess > 0:
            w[int(rng.integers(0, spec.N))] += excess
    else:
        raise ValueError("initial must be 'random', 'exponential', or 'precondensed'")
    rng.shuffle(w)
    if int(w.sum()) != spec.M:
        raise RuntimeError("failed to initialize exact total wealth")
    return w


def _seed_for(spec: Spec, segment_index: int) -> int:
    return int((spec.seed + 1_000_003 * segment_index + 97_409) % (2**32 - 1))


def _advance(w: np.ndarray, order: np.ndarray, sweeps: int, spec: Spec, seg: int) -> None:
    if spec.bias_kind != "step":
        raise ValueError("publication study is defined for step bias; use bias_kind='step'")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for publication-scale runs")
    _advance_builtin(w, order, sweeps, 0, spec.c, spec.x0, _seed_for(spec, seg))


def _topk_desc(w: np.ndarray, k: int) -> np.ndarray:
    k = min(k, w.size)
    if k == w.size:
        return np.sort(w)[::-1].astype(float)
    idx = np.argpartition(w, w.size - k)[-k:]
    return np.sort(w[idx])[::-1].astype(float)


def _ensure_hist(hist: np.ndarray, n: int) -> np.ndarray:
    if n <= hist.size:
        return hist
    return np.pad(hist, (0, n - hist.size))


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **state)
    # A sync client or scanner can briefly hold the destination open on Windows.
    for attempt in range(7):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.1 * 2**attempt)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        # Burn-in and production rows intentionally carry different diagnostics.
        # Preserve a stable first-seen order while taking the union of all keys.
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key); fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(spec_path: Path, force: bool = False) -> int:
    spec = _load_spec(spec_path)
    out = Path(spec.output_directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "publication_checkpoint.npz"
    fingerprint = _fingerprint(spec)

    if spec.block_sweeps <= 0 or spec.sample_every <= 0 or spec.order_every <= 0:
        raise ValueError("sweep intervals must be positive")
    if spec.block_sweeps % spec.sample_every or spec.block_sweeps % spec.order_every:
        raise ValueError("sample_every and order_every must divide block_sweeps")
    if spec.burnin_sweeps % spec.block_sweeps or spec.production_sweeps % spec.block_sweeps:
        raise ValueError("burnin_sweeps and production_sweeps must be multiples of block_sweeps")

    if checkpoint.exists() and not force:
        with np.load(checkpoint, allow_pickle=False) as d:
            if str(d["fingerprint"]) != fingerprint:
                raise RuntimeError(f"checkpoint fingerprint mismatch in {out}")
            w = d["wealth"].astype(np.int64)
            burn_done = int(d["burn_done"])
            prod_done = int(d["prod_done"])
            segment_index = int(d["segment_index"])
            hist = d["hist"].astype(np.int64)
            sample_count = int(d["sample_count"])
            block_records = [json.loads(x) for x in d["block_records"].tolist()]
            order_records = [json.loads(x) for x in d["order_records"].tolist()]
        print(f"Resuming {spec.run_id}: burn={burn_done:,}, production={prod_done:,}", flush=True)
    else:
        w = _initialize(spec)
        burn_done = 0
        prod_done = 0
        segment_index = 0
        hist = np.zeros(1, dtype=np.int64)
        sample_count = 0
        block_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []

    order = np.arange(spec.N, dtype=np.int64)
    t0 = time.monotonic()
    blocks_since_checkpoint = 0

    def checkpoint_now() -> None:
        state = {
            "fingerprint": np.asarray(fingerprint),
            "wealth": w,
            "burn_done": np.asarray(burn_done),
            "prod_done": np.asarray(prod_done),
            "segment_index": np.asarray(segment_index),
            "hist": hist,
            "sample_count": np.asarray(sample_count),
            "block_records": np.asarray([json.dumps(x) for x in block_records]),
            "order_records": np.asarray([json.dumps(x) for x in order_records]),
        }
        _save_checkpoint(checkpoint, state)

    # Fixed burn-in.  We still record one snapshot per block to expose slow
    # relaxation in p0, Y2, and the top order statistics.
    while burn_done < spec.burnin_sweeps:
        _advance(w, order, spec.block_sweeps, spec, segment_index)
        segment_index += 1
        burn_done += spec.block_sweeps
        top = _topk_desc(w, min(spec.top_k, 5))
        y2 = float(np.dot(w.astype(np.float64), w.astype(np.float64)) / max(spec.M**2, 1))
        rec = {
            "phase": "burnin", "sweep": burn_done, "p0": float(np.mean(w == 0)),
            "Y2": y2, "max_fraction": float(w.max() / max(spec.M, 1)),
        }
        for i, value in enumerate(top, 1):
            rec[f"m{i}"] = float(value)
        if len(top) >= 2 and top[1] > 0:
            rec["m1_over_m2"] = float(top[0] / top[1])
            if len(top) >= 3:
                rec["m3_over_m2"] = float(top[2] / top[1])
        block_records.append(rec)
        blocks_since_checkpoint += 1
        if blocks_since_checkpoint >= spec.checkpoint_every_blocks:
            checkpoint_now(); blocks_since_checkpoint = 0
        print(f"{spec.run_id}: burn {burn_done:,}/{spec.burnin_sweeps:,}, p0={rec['p0']:.6f}, maxfrac={rec['max_fraction']:.4g}", flush=True)

    # Production sampling.  Histogram/p0/Y2 are sampled at sample_every;
    # order statistics are sampled at the typically sparser order_every.
    while prod_done < spec.production_sweeps:
        block_p0: list[float] = []
        block_y2: list[float] = []
        block_max: list[float] = []
        block_top: list[np.ndarray] = []
        within = 0
        next_order = spec.order_every
        while within < spec.block_sweeps:
            step = min(spec.sample_every, spec.block_sweeps - within)
            _advance(w, order, step, spec, segment_index)
            segment_index += 1
            within += step
            prod_done += step

            bc = np.bincount(w)
            hist = _ensure_hist(hist, bc.size)
            hist[:bc.size] += bc
            sample_count += 1
            p0 = float(np.mean(w == 0))
            wf = w.astype(np.float64)
            y2 = float(np.dot(wf, wf) / max(spec.M**2, 1))
            maxfrac = float(w.max() / max(spec.M, 1))
            block_p0.append(p0); block_y2.append(y2); block_max.append(maxfrac)

            # order_every is a multiple of sample_every; record when crossed.
            if within >= next_order:
                top = _topk_desc(w, spec.top_k)
                row = {
                    "production_sweep": prod_done,
                    "total_sweep": spec.burnin_sweeps + prod_done,
                    "p0": p0, "Y2": y2, "max_fraction": maxfrac,
                }
                for i, value in enumerate(top, 1):
                    row[f"m{i}"] = float(value)
                order_records.append(row)
                block_top.append(top)
                next_order += spec.order_every

        rec = {
            "phase": "production",
            "sweep": spec.burnin_sweeps + prod_done,
            "production_sweep": prod_done,
            "p0": float(np.mean(block_p0)),
            "p0_sd": float(np.std(block_p0, ddof=1)) if len(block_p0) > 1 else 0.0,
            "Y2": float(np.mean(block_y2)),
            "Y2_sd": float(np.std(block_y2, ddof=1)) if len(block_y2) > 1 else 0.0,
            "max_fraction": float(np.mean(block_max)),
            "max_fraction_sd": float(np.std(block_max, ddof=1)) if len(block_max) > 1 else 0.0,
        }
        if block_top:
            arr = np.vstack(block_top)
            for i in range(min(spec.top_k, arr.shape[1])):
                rec[f"m{i+1}"] = float(arr[:, i].mean())
            if arr.shape[1] >= 2:
                ratio12 = arr[:, 0] / np.maximum(arr[:, 1], 1.0)
                rec["m1_over_m2"] = float(np.mean(ratio12))
                if arr.shape[1] >= 3:
                    ratio32 = arr[:, 2] / np.maximum(arr[:, 1], 1.0)
                    rec["m3_over_m2"] = float(np.mean(ratio32))
        block_records.append(rec)
        blocks_since_checkpoint += 1
        if blocks_since_checkpoint >= spec.checkpoint_every_blocks:
            checkpoint_now(); blocks_since_checkpoint = 0
        print(f"{spec.run_id}: production {prod_done:,}/{spec.production_sweeps:,}, p0={rec['p0']:.6f}, Y2={rec['Y2']:.4g}, maxfrac={rec['max_fraction']:.4g}", flush=True)

    checkpoint_now()
    elapsed = time.monotonic() - t0

    prob = hist / hist.sum() if hist.sum() else np.zeros_like(hist, dtype=float)
    ccdf = np.cumsum(prob[::-1])[::-1]
    _write_csv(out / "block_metrics.csv", block_records)
    _write_csv(out / "order_samples.csv", order_records)
    with (out / "distribution.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["m", "count", "p", "S"])
        for m in range(prob.size):
            writer.writerow([m, int(hist[m]), float(prob[m]), float(ccdf[m])])

    prod_blocks = [r for r in block_records if r.get("phase") == "production"]
    summary = {
        "run_id": spec.run_id, "study": spec.study, "N": spec.N, "M": spec.M,
        "mu": spec.mu, "seed": spec.seed, "c": spec.c,
        "burnin_sweeps": spec.burnin_sweeps, "production_sweeps": spec.production_sweeps,
        "sample_count": sample_count, "elapsed_seconds_this_invocation": elapsed,
        "p0_mean": float(np.mean([r["p0"] for r in prod_blocks])) if prod_blocks else None,
        "p0_block_sd": float(np.std([r["p0"] for r in prod_blocks], ddof=1)) if len(prod_blocks) > 1 else None,
        "Y2_mean": float(np.mean([r["Y2"] for r in prod_blocks])) if prod_blocks else None,
        "max_fraction_mean": float(np.mean([r["max_fraction"] for r in prod_blocks])) if prod_blocks else None,
        "final_top": _topk_desc(w, min(spec.top_k, 10)).tolist(),
        "mu_c_reference": spec.mu_c_reference, "mu_ratio": spec.mu_ratio,
        "notes": spec.notes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "run_spec.json").write_text(json.dumps(asdict(spec), indent=2))
    print(f"DONE {spec.run_id}: {out}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("spec", type=Path)
    p.add_argument("--force", action="store_true", help="ignore an existing checkpoint and start over")
    a = p.parse_args()
    return run(a.spec, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
