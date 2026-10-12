"""Benchmark common tabular file formats with local hardware metrics."""

from __future__ import annotations

import argparse
import gc
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import tzdata


DEFAULT_ROWS = 500_000
CPU_TDP_WATTS = 65.0
BYTES_PER_MB = 1024**2


@dataclass(frozen=True)
class BenchmarkResult:
    """Measurements collected for one file format."""

    format_name: str
    file_size_mb: float
    write_seconds: float
    read_seconds: float
    peak_memory_mb: float
    cpu_seconds: float
    energy_wh: float
    size_savings_percent: float = 0.0
    write_savings_percent: float = 0.0
    read_savings_percent: float = 0.0
    memory_savings_percent: float = 0.0
    cpu_savings_percent: float = 0.0
    energy_savings_percent: float = 0.0


def create_benchmark_dataframe(rows: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Create deterministic benchmark data with the requested six columns."""
    if rows <= 0:
        raise ValueError("rows must be greater than zero")

    index = np.arange(rows, dtype=np.int64)
    categories = np.array(["Electronics", "Home", "Books", "Sports", "Clothing"])
    dates = pd.date_range("2020-01-01", periods=365, freq="D")

    return pd.DataFrame(
        {
            "id": index,
            "name": np.char.add("Customer ", index.astype(str)),
            "email": np.char.add(np.char.add("customer", index.astype(str)), "@example.com"),
            "amount": np.round(10.0 + (index % 10000) / 100.0, 2),
            "date": dates[index % len(dates)],
            "category": categories[index % len(categories)],
        }
    )


def _write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_csv(path, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _write_xlsx(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_excel(path, index=False, engine="openpyxl")


def _read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", parse_dates=["date"])


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_parquet(path, index=False, engine="fastparquet")


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="fastparquet")


def _write_orc(dataframe: pd.DataFrame, path: Path) -> None:
    _configure_orc_timezone_database()
    orc.write_table(pa.Table.from_pandas(dataframe, preserve_index=False), path)


def _read_orc(path: Path) -> pd.DataFrame:
    _configure_orc_timezone_database()
    return orc.read_table(path).to_pandas()


def _write_feather(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_feather(path)


def _read_feather(path: Path) -> pd.DataFrame:
    return pd.read_feather(path)


FORMAT_HANDLERS: dict[str, tuple[str, Callable[[pd.DataFrame, Path], None], Callable[[Path], pd.DataFrame]]] = {
    "CSV": (".csv", _write_csv, _read_csv),
    "XLSX": (".xlsx", _write_xlsx, _read_xlsx),
    "Parquet": (".parquet", _write_parquet, _read_parquet),
    "ORC": (".orc", _write_orc, _read_orc),
    "Feather": (".feather", _write_feather, _read_feather),
}


def _configure_orc_timezone_database() -> None:
    """Use the bundled IANA database because Windows has no /usr/share/zoneinfo."""
    timezone_dir = Path(tzdata.__file__).parent / "zoneinfo"
    if not timezone_dir.exists():
        raise RuntimeError(f"tzdata zoneinfo directory not found: {timezone_dir}")
    os.environ.setdefault("TZDIR", str(timezone_dir))


def _percent_savings(baseline: float, value: float) -> float:
    """Return reduction relative to baseline, or zero when baseline is zero."""
    if baseline == 0:
        return 0.0
    return (baseline - value) / baseline * 100.0


def benchmark_format(dataframe: pd.DataFrame, format_name: str, output_dir: Path) -> BenchmarkResult:
    """Write and read one format while measuring time, CPU, memory, and size."""
    try:
        extension, writer, reader = FORMAT_HANDLERS[format_name]
    except KeyError as error:
        raise ValueError(f"Unsupported format: {format_name}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark{extension}"
    if path.exists():
        path.unlink()

    tracemalloc.start()
    tracemalloc.reset_peak()
    cpu_start = time.process_time()

    write_start = time.perf_counter()
    writer(dataframe, path)
    write_seconds = time.perf_counter() - write_start

    read_start = time.perf_counter()
    loaded = reader(path)
    read_seconds = time.perf_counter() - read_start

    cpu_seconds = time.process_time() - cpu_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del loaded
    gc.collect()

    return BenchmarkResult(
        format_name=format_name,
        file_size_mb=path.stat().st_size / BYTES_PER_MB,
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        peak_memory_mb=peak_bytes / BYTES_PER_MB,
        cpu_seconds=cpu_seconds,
        energy_wh=cpu_seconds * CPU_TDP_WATTS / 3600.0,
    )


def add_csv_baseline_savings(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Add percentage reductions for every metric relative to the CSV result."""
    csv_result = next((result for result in results if result.format_name == "CSV"), None)
    if csv_result is None:
        raise ValueError("CSV is required as the baseline format")

    return [
        BenchmarkResult(
            **{
                **result.__dict__,
                "size_savings_percent": _percent_savings(csv_result.file_size_mb, result.file_size_mb),
                "write_savings_percent": _percent_savings(csv_result.write_seconds, result.write_seconds),
                "read_savings_percent": _percent_savings(csv_result.read_seconds, result.read_seconds),
                "memory_savings_percent": _percent_savings(csv_result.peak_memory_mb, result.peak_memory_mb),
                "cpu_savings_percent": _percent_savings(csv_result.cpu_seconds, result.cpu_seconds),
                "energy_savings_percent": _percent_savings(csv_result.energy_wh, result.energy_wh),
            }
        )
        for result in results
    ]


def print_results(results: list[BenchmarkResult], rows: int) -> None:
    """Print a formatted comparison table and its measurement assumptions."""
    columns = [
        ("Format", "format_name"),
        ("Size MB", "file_size_mb"),
        ("Write s", "write_seconds"),
        ("Read s", "read_seconds"),
        ("Peak MB", "peak_memory_mb"),
        ("CPU s", "cpu_seconds"),
        ("Energy Wh", "energy_wh"),
        ("Size save %", "size_savings_percent"),
        ("Write save %", "write_savings_percent"),
        ("Read save %", "read_savings_percent"),
        ("Mem save %", "memory_savings_percent"),
        ("CPU save %", "cpu_savings_percent"),
        ("Energy save %", "energy_savings_percent"),
    ]
    headers = [header for header, _ in columns]
    rows_as_strings = [
        [
            getattr(result, attribute) if attribute == "format_name" else f"{getattr(result, attribute):.2f}"
            for _, attribute in columns
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(str(row[index])) for row in rows_as_strings)) for index, header in enumerate(headers)]
    print(f"\nFormat benchmark ({rows:,} rows, CPU TDP assumption: {CPU_TDP_WATTS:.0f} W)")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows_as_strings:
        print(" | ".join(str(value).rjust(widths[index]) if index else str(value).ljust(widths[index]) for index, value in enumerate(row)))
    print("\nSavings are reductions versus CSV; negative values mean the format used more resources.")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "format_benchmark")
    return parser.parse_args()


def main() -> None:
    """Run the complete benchmark."""
    args = parse_args()
    dataframe = create_benchmark_dataframe(args.rows)
    results = [
        benchmark_format(dataframe, format_name, args.output_dir)
        for format_name in FORMAT_HANDLERS
    ]
    print_results(add_csv_baseline_savings(results), args.rows)


if __name__ == "__main__":
    main()
