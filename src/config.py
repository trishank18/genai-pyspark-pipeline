"""Shared paths, defaults, and logging configuration for the project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Store project paths and default record counts in one place."""

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    customer_count: int = 100_000
    product_count: int = 10_000
    order_count: int = 1_000_000
    random_seed: int = 42


def configure_logging(level: int = logging.INFO) -> None:
    """Configure readable console logging once for command-line programs.

    Args:
        level: The minimum severity level to display.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_data_directories(settings: Settings) -> None:
    """Create the project's raw and processed data directories when absent.

    Args:
        settings: Project path settings to use.
    """
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
