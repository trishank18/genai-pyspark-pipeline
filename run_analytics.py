"""Run the sales analytics against the generated Parquet datasets."""

from __future__ import annotations

import time
from pathlib import Path

from src.spark_analytics import SalesAnalytics, create_spark_session


def run() -> None:
    """Load the raw datasets, display analytics, and report execution times."""
    raw_data_dir = Path(__file__).resolve().parent / "data" / "raw"
    spark = create_spark_session()

    try:
        customers = spark.read.parquet(str(raw_data_dir / "customers.parquet"))
        products = spark.read.parquet(str(raw_data_dir / "products.parquet"))
        orders = spark.read.parquet(str(raw_data_dir / "orders.parquet"))
        analytics = SalesAnalytics(customers, products, orders)

        operations = (
            ("Top customers", analytics.top_customers),
            ("Sales by category", analytics.sales_by_category),
            ("Monthly trends", analytics.monthly_trends),
        )
        for name, operation in operations:
            start = time.perf_counter()
            print(f"\n{name}:")
            operation().show()
            elapsed = time.perf_counter() - start
            print(f"{name} execution time: {elapsed:.2f} seconds")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
