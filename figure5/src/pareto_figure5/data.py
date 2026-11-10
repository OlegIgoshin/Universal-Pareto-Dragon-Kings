"""Download and read the UN population workbook used for Figure 5."""

from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


WORKBOOK_NAME = "WUP2025-F21-DEGURBA-Cities_Pop.xlsx"
SOURCE_URL = f"https://population.un.org/wup/assets/Download/Cities/{WORKBOOK_NAME}"
REQUIRED_COLUMNS = ("Location", "ISO3_Code", "City_Name", "2025")


@dataclass(frozen=True)
class City:
    country: str
    name: str
    population: int


def _data_rows(book):
    if "Data" not in book.sheetnames:
        raise ValueError("The UN workbook must contain a 'Data' worksheet.")
    return book["Data"].iter_rows(values_only=True)


def _column_indices(header):
    columns = {str(name).strip(): index for index, name in enumerate(header)}
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"Missing UN workbook columns: {', '.join(missing)}.")
    return tuple(columns[name] for name in REQUIRED_COLUMNS)


def download_workbook(cache_dir: Path, refresh: bool = False) -> Path:
    """Return the cached File 21 workbook, downloading it when needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    workbook = cache_dir / WORKBOOK_NAME
    if workbook.is_file() and not refresh:
        return workbook

    request = Request(SOURCE_URL, headers={"User-Agent": "pareto-figure5/1.0"})
    temporary = None
    try:
        with NamedTemporaryFile(dir=cache_dir, suffix=".xlsx", delete=False) as file:
            temporary = Path(file.name)
            with urlopen(request, timeout=60) as response:
                copyfileobj(response, file)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise BadZipFile("The downloaded workbook failed its integrity check.")
        with closing(load_workbook(temporary, read_only=True, data_only=True)) as book:
            _column_indices(next(_data_rows(book), ()))
        temporary.replace(workbook)
    except (OSError, HTTPException, BadZipFile, ValueError, InvalidFileException,
            ParseError, KeyError) as exc:
        raise RuntimeError(f"Could not download the UN workbook from {SOURCE_URL}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return workbook


def load_cities(workbook: Path) -> dict[str, list[City]]:
    """Read 2025 populations of at least 50,000, grouped by UN ISO3 code."""
    cities_by_country = defaultdict(list)
    with closing(load_workbook(workbook, read_only=True, data_only=True)) as book:
        rows = _data_rows(book)
        country_col, iso3_col, name_col, population_col = _column_indices(next(rows, ()))
        for row_number, row in enumerate(rows, start=2):
            population_thousands = row[population_col]
            if not isinstance(population_thousands, (int, float)) or population_thousands < 50:
                continue
            iso3 = row[iso3_col]
            country = row[country_col]
            name = row[name_col]
            if not all(isinstance(value, str) and value.strip() for value in (iso3, country, name)):
                raise ValueError(f"Missing country or city identifiers in workbook row {row_number}.")
            cities_by_country[iso3].append(City(country, name, round(population_thousands * 1000)))

    if not cities_by_country:
        raise ValueError("The UN workbook contains no eligible 2025 city populations.")
    return {
        iso3: sorted(cities, key=lambda city: -city.population)
        for iso3, cities in cities_by_country.items()
    }
