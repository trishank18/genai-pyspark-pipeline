"""Flask web application for the sales analytics dashboard."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from compare_performance import benchmark_pyspark


ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
app = Flask(__name__)


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache the source Parquet files for the application process."""
    return (
        pd.read_parquet(RAW_DATA_DIR / "orders.parquet"),
        pd.read_parquet(RAW_DATA_DIR / "products.parquet"),
    )


@lru_cache(maxsize=1)
def build_analytics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build customer, category, and monthly summaries once per process."""
    orders, products = load_data()
    enriched = orders.merge(products[["product_id", "category", "price"]], on="product_id", how="inner")
    enriched["revenue"] = enriched["quantity"] * enriched["price"]

    customers = (
        enriched.groupby("customer_id", as_index=False)
        .agg(lifetime_value=("revenue", "sum"), order_count=("order_id", "count"))
        .sort_values("lifetime_value", ascending=False)
    )
    categories = (
        enriched.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("quantity", "sum"), order_count=("order_id", "count"))
        .sort_values("revenue", ascending=False)
    )
    dated = enriched.assign(order_date=pd.to_datetime(enriched["order_date"]))
    monthly = (
        dated.assign(month=dated["order_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"), order_count=("order_id", "count"))
        .sort_values("month")
    )
    return customers, categories, monthly


def _records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-safe records."""
    return dataframe.astype(object).where(dataframe.notna(), None).to_dict(orient="records")


def _benchmark_pandas() -> dict[str, float | str]:
    """Time the Pandas pipeline for the performance view."""
    orders, products = load_data()
    load_start = time.perf_counter()
    orders = orders.copy()
    products = products.copy()
    load_seconds = time.perf_counter() - load_start

    join_start = time.perf_counter()
    enriched = orders.merge(products[["product_id", "price"]], on="product_id", how="inner")
    enriched["revenue"] = enriched["quantity"] * enriched["price"]
    join_seconds = time.perf_counter() - join_start

    aggregation_start = time.perf_counter()
    enriched.groupby("customer_id", as_index=False)["revenue"].sum().nlargest(10, "revenue")
    aggregation_seconds = time.perf_counter() - aggregation_start
    return {
        "Engine": "Pandas",
        "Load (s)": load_seconds,
        "Join + revenue (s)": join_seconds,
        "Group + top 10 (s)": aggregation_seconds,
        "Total (s)": load_seconds + join_seconds + aggregation_seconds,
    }


@app.get("/")
def index():
    """Render the dashboard shell."""
    return render_template("index.html")


@app.get("/api/dashboard")
def dashboard_data():
    """Return filtered analytics data for the dashboard."""
    customers, categories, monthly = build_analytics()
    requested_top = request.args.get("top", "20")
    try:
        top = max(1, min(int(requested_top), len(customers)))
    except ValueError:
        top = 20
    search = request.args.get("search", "").strip().lower()
    if search:
        customers = customers[customers["customer_id"].astype(str).str.contains(search, regex=False)]

    orders, products = load_data()
    total_revenue = float(categories["revenue"].sum())
    total_orders = int(len(orders))
    return jsonify(
        {
            "metrics": {
                "orders": total_orders,
                "products": int(len(products)),
                "revenue": total_revenue,
                "average_order": total_revenue / total_orders if total_orders else 0,
            },
            "top_customers": _records(customers.head(top)),
            "customer_count": int(len(customers)),
            "categories": _records(categories),
            "monthly": _records(monthly),
        }
    )


@app.get("/api/benchmark")
def performance_data():
    """Run and return Pandas and PySpark performance measurements."""
    pandas_result = _benchmark_pandas()
    spark = _spark_session()
    try:
        spark_result = benchmark_pyspark(
            spark,
            RAW_DATA_DIR / "orders.parquet",
            RAW_DATA_DIR / "products.parquet",
        )
    finally:
        spark.stop()
    return jsonify(
        {
            "results": [
                pandas_result,
                {
                    "Engine": spark_result.engine,
                    "Load (s)": spark_result.load_seconds,
                    "Join + revenue (s)": spark_result.join_revenue_seconds,
                    "Group + top 10 (s)": spark_result.aggregation_seconds,
                    "Total (s)": spark_result.total_seconds,
                },
            ]
        }
    )


def _spark_session():
    """Create a local Spark session for the on-demand benchmark."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("FlaskPandasVsPySpark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
