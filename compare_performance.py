"""Compare Pandas and PySpark performance for a 1M-row sales aggregation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing measurements for one analytics engine."""

    engine: str
    load_seconds: float
    join_revenue_seconds: float
    aggregation_seconds: float

    @property
    def total_seconds(self) -> float:
        """Return the elapsed time for all measured operations."""
        return self.load_seconds + self.join_revenue_seconds + self.aggregation_seconds


def _data_paths() -> tuple[Path, Path]:
    """Return the orders and products Parquet paths."""
    raw_data_dir = Path(__file__).resolve().parent / "data" / "raw"
    return raw_data_dir / "orders.parquet", raw_data_dir / "products.parquet"


def benchmark_pandas(orders_path: Path, products_path: Path) -> BenchmarkResult:
    """Benchmark loading, joining/revenue calculation, and top-10 aggregation in Pandas."""
    load_start = time.perf_counter()
    orders = pd.read_parquet(orders_path)
    products = pd.read_parquet(products_path)
    load_seconds = time.perf_counter() - load_start

    join_start = time.perf_counter()
    enriched = orders.merge(products[["product_id", "price"]], on="product_id", how="inner")
    enriched["revenue"] = enriched["quantity"] * enriched["price"]
    join_revenue_seconds = time.perf_counter() - join_start

    aggregation_start = time.perf_counter()
    top_customers = (
        enriched.groupby("customer_id", as_index=False)["revenue"]
        .sum()
        .nlargest(10, "revenue")
        .reset_index(drop=True)
    )
    aggregation_seconds = time.perf_counter() - aggregation_start
    print("\nPandas top 10:")
    print(top_customers.to_string(index=False))

    return BenchmarkResult("Pandas", load_seconds, join_revenue_seconds, aggregation_seconds)


def benchmark_pyspark(
    spark: SparkSession, orders_path: Path, products_path: Path
) -> BenchmarkResult:
    """Benchmark the equivalent operations in PySpark."""
    load_start = time.perf_counter()
    orders = spark.read.parquet(str(orders_path))
    products = spark.read.parquet(str(products_path))
    # Spark reads lazily, so count forces the Parquet load to be measured here.
    orders.count()
    products.count()
    load_seconds = time.perf_counter() - load_start

    join_start = time.perf_counter()
    enriched = (
        orders.join(products.select("product_id", "price"), on="product_id", how="inner")
        .withColumn("revenue", F.col("quantity") * F.col("price"))
    )
    enriched = enriched.cache()
    enriched.count()
    join_revenue_seconds = time.perf_counter() - join_start

    aggregation_start = time.perf_counter()
    top_customers = (
        enriched.groupBy("customer_id")
        .agg(F.sum("revenue").alias("revenue"))
        .orderBy(F.desc("revenue"))
        .limit(10)
    )
    top_customers.show(truncate=False)
    aggregation_seconds = time.perf_counter() - aggregation_start
    enriched.unpersist()

    return BenchmarkResult("PySpark", load_seconds, join_revenue_seconds, aggregation_seconds)


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print benchmark timings in a readable comparison table."""
    headers = ("Engine", "Load (s)", "Join + revenue (s)", "Group + top 10 (s)", "Total (s)")
    rows = [
        (
            result.engine,
            f"{result.load_seconds:.2f}",
            f"{result.join_revenue_seconds:.2f}",
            f"{result.aggregation_seconds:.2f}",
            f"{result.total_seconds:.2f}",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    print("\nPerformance comparison")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    """Run both benchmarks against the generated Parquet files."""
    orders_path, products_path = _data_paths()
    pandas_result = benchmark_pandas(orders_path, products_path)

    spark = (
        SparkSession.builder.appName("PandasVsPySparkPerformance")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        pyspark_result = benchmark_pyspark(spark, orders_path, products_path)
    finally:
        spark.stop()

    print_comparison([pandas_result, pyspark_result])


if __name__ == "__main__":
    main()
