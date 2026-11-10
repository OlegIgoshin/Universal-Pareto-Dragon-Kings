"""Command-line entry point for Figure 5 reproduction."""

import argparse
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

try:
    from openpyxl.utils.exceptions import InvalidFileException
except ImportError as exc:
    raise SystemExit(
        "Figure 5 requires openpyxl. Install all dependencies with:\n"
        "  python -m pip install -r requirements.txt\n"
        "Run this command from the top-level source package directory using "
        "the same Python executable that runs run_all.py."
    ) from exc

from .analysis import analyze
from .checks import supporting_checks
from .data import download_workbook, load_cities
from .output import write_results
from .plot import write_figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figure5_output"),
        help="directory for figures, tables, and statistics (default: figure5_output)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".figure5_cache"),
        help="directory for the official UN workbook (default: .figure5_cache)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workbook", type=Path, help="read an existing UN File 21 workbook")
    source.add_argument("--refresh", action="store_true", help="download the UN workbook again")
    parser.add_argument(
        "--rank-limit", "--K", type=int, default=40, metavar="K",
        help="minimum qualifying cities per country AND last analyzed rank (default: 40)",
    )
    args = parser.parse_args(argv)

    try:
        workbook = args.workbook
        if workbook is None:
            workbook = download_workbook(args.cache_dir, refresh=args.refresh)
        result = analyze(load_cities(workbook), rank_limit=args.rank_limit)
        checks = supporting_checks(result)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_figure(result, args.output_dir)
        write_results(result, checks, workbook, args.output_dir)
    except (OSError, ValueError, RuntimeError, BadZipFile, InvalidFileException, ParseError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")

    for label, group in (("All countries", result.all_countries), ("Screened countries", result.screened_countries)):
        print(f"{label}: {group.above_median}/{group.n} above median (p={group.p_value:.6g})")
    print(f"Figure and statistics saved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
