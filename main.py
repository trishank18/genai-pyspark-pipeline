"""Generate and save the synthetic e-commerce datasets."""

from __future__ import annotations

import time
import logging
from pathlib import Path

from src import config
from src.data_generator import SyntheticDataGenerator


def _file_size(path: Path) -> str:
    """Return a human-readable size for a generated file."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main() -> None:
    """Generate all datasets and save them as Parquet files."""
    start_time = time.perf_counter()

    try:
        settings = config.Settings()
        config.configure_logging()
        config.create_data_directories(settings)
        generator = SyntheticDataGenerator(
            customer_count=settings.customer_count,
            product_count=settings.product_count,
            order_count=settings.order_count,
            seed=settings.random_seed,
        )
        customers, products, orders = generator.generate_all()

        datasets = {
            "customers": customers,
            "products": products,
            "orders": orders,
        }
        saved_files: list[tuple[str, str]] = []
        for name, dataframe in datasets.items():
            destination = settings.raw_data_dir / f"{name}.parquet"
            dataframe.to_parquet(destination, index=False)
            saved_files.append((destination.name, _file_size(destination)))

        elapsed = time.perf_counter() - start_time
        print(f"\nCompleted in {elapsed:.1f} seconds")
        print("\nFiles saved:")
        for filename, size in saved_files:
            print(f"- {filename} ({size})")
    except Exception as error:
        elapsed = time.perf_counter() - start_time
        logging.error("Generation failed after %.1f seconds: %s", elapsed, error)
        raise


if __name__ == "__main__":
    main()