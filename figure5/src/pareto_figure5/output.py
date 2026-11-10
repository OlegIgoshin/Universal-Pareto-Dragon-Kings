"""Export the figures' underlying values and analysis provenance."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .analysis import FIT_THRESHOLD, Figure5Analysis
from .data import SOURCE_URL


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_results(
    result: Figure5Analysis, checks: dict[str, object], workbook: Path, output_dir: Path
) -> None:
    _write_csv(
        output_dir / "city_ranks.csv",
        ("iso3", "country", "rank", "city", "population", "population_over_mK"),
        (
            (c.iso3, c.country, rank, city.name, city.population, x)
            for c in result.countries
            for rank, city, x in zip(result.ranks, c.cities, c.normalized)
        ),
    )
    _write_csv(
        output_dir / "country_statistics.csv",
        (
            "iso3", "country", "qualifying_cities", "leading_city", "leading_population",
            "reference_population", "leading_over_reference", "background_gap_sum",
            "background_ks", "background_p", "passes_background", "dk_r", "dk_p", "above_median",
        ),
        (
            (
                c.iso3, c.country, c.qualifying_cities, c.cities[0].name, c.cities[0].population,
                c.reference_population, c.cities[0].population / c.reference_population,
                c.background_sum, c.background_ks, c.background_p, c.passes_background,
                c.dk_r, c.dk_p, c.dk_p < 0.5,
            )
            for c in result.countries
        ),
    )
    edges = result.density_edges
    _write_csv(
        output_dir / "pooled_density.csv",
        ("bin_left", "bin_right", "bin_center", "density"),
        zip(edges[:-1], edges[1:], result.density_centers, result.density),
    )
    ranks = result.ranks
    _write_csv(
        output_dir / "aggregate_rank_profile.csv",
        ("rank", "geometric_mean", "country_q25", "country_q75", "zipf_reference", "background_fit"),
        zip(
            ranks, result.rank_geomean, result.rank_q25, result.rank_q75,
            len(ranks) / ranks, result.rank_amplitude * ranks**result.rank_slope,
        ),
    )

    stats = {
        "workbook": {
            "filename": workbook.name,
            "official_url": SOURCE_URL,
            "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        },
        "selection": {
            "population_year": 2025,
            "minimum_population": 50_000,
            "ranks_per_country": len(ranks),
            "background_p_minimum": FIT_THRESHOLD,
        },
        "panel_a": {
            "pooled_survival_exponent": result.pooled_q,
            "pooled_density_exponent": -(1 + result.pooled_q),
            "product_limit_mass_in_bins": float(result.density @ np.diff(edges)),
        },
        "panel_b": {
            "rank_exponent": result.rank_slope,
            "rank_amplitude": result.rank_amplitude,
            "rank1_geometric_mean": float(result.rank_geomean[0]),
        },
        "panel_d": {
            "all_countries": asdict(result.all_countries),
            "screened_countries": asdict(result.screened_countries),
        },
        "supporting_checks": checks,
    }
    (output_dir / "figure5_statistics.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
