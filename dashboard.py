"""Streamlit dashboard for sales analytics and engine performance."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from compare_performance import benchmark_pyspark


st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

RAW_DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the orders and products used by the dashboard."""
    orders = pd.read_parquet(RAW_DATA_DIR / "orders.parquet")
    products = pd.read_parquet(RAW_DATA_DIR / "products.parquet")
    return orders, products


@st.cache_data
def build_analytics(orders: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build customer, category, and monthly sales summaries."""
    enriched = orders.merge(products[["product_id", "category", "price"]], on="product_id", how="inner")
    enriched["revenue"] = enriched["quantity"] * enriched["price"]

    customer_summary = (
        enriched.groupby("customer_id", as_index=False)
        .agg(lifetime_value=("revenue", "sum"), order_count=("order_id", "count"))
        .sort_values("lifetime_value", ascending=False)
    )
    sales_by_category = (
        enriched.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("quantity", "sum"), order_count=("order_id", "count"))
        .sort_values("revenue", ascending=False)
    )
    orders_with_dates = enriched.assign(order_date=pd.to_datetime(enriched["order_date"]))
    monthly_trends = (
        orders_with_dates.assign(month=orders_with_dates["order_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"), order_count=("order_id", "count"))
        .sort_values("month")
    )
    return customer_summary, sales_by_category, monthly_trends


@st.cache_data
def benchmark_pandas(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Measure the Pandas load, join, and aggregation stages."""
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

    return pd.DataFrame(
        [
            {
                "Engine": "Pandas",
                "Load (s)": load_seconds,
                "Join + revenue (s)": join_seconds,
                "Group + top 10 (s)": aggregation_seconds,
            }
        ]
    ).assign(**{"Total (s)": lambda frame: frame.iloc[:, 1:].sum(axis=1)})


def benchmark_pyspark_from_paths() -> pd.DataFrame:
    """Run the PySpark benchmark and return its timings as a table."""
    spark = (
        SparkSession.builder.appName("DashboardPandasVsPySpark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        result = benchmark_pyspark(
            spark,
            RAW_DATA_DIR / "orders.parquet",
            RAW_DATA_DIR / "products.parquet",
        )
    finally:
        spark.stop()
    return pd.DataFrame(
        [
            {
                "Engine": result.engine,
                "Load (s)": result.load_seconds,
                "Join + revenue (s)": result.join_revenue_seconds,
                "Group + top 10 (s)": result.aggregation_seconds,
                "Total (s)": result.total_seconds,
            }
        ]
    )


def format_currency(value: float) -> str:
    """Format a revenue value for metric cards."""
    return f"${value:,.2f}"


st.title("📊 Sales Analytics Dashboard")
st.caption("Interactive view of the generated 1M-row e-commerce dataset.")

try:
    orders, products = load_data()
    top_customers, sales_by_category, monthly_trends = build_analytics(orders, products)
except FileNotFoundError as error:
    st.error(f"Required Parquet data is missing: {error.filename}")
    st.stop()

total_revenue = float(sales_by_category["revenue"].sum())
total_orders = int(len(orders))
average_order_value = total_revenue / total_orders if total_orders else 0.0

metric_columns = st.columns(4)
metric_columns[0].metric("Orders", f"{total_orders:,}")
metric_columns[1].metric("Products", f"{len(products):,}")
metric_columns[2].metric("Total revenue", format_currency(total_revenue))
metric_columns[3].metric("Average order value", format_currency(average_order_value))

overview_tab, customers_tab, categories_tab, trends_tab, performance_tab = st.tabs(
    ["Overview", "Top customers", "Categories", "Monthly trends", "Performance"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Revenue by category")
        st.bar_chart(sales_by_category.set_index("category")["revenue"])
    with right:
        st.subheader("Monthly revenue")
        st.line_chart(monthly_trends.set_index("month")["revenue"])
    st.subheader("Top 10 customers by lifetime value")
    st.dataframe(top_customers.head(10), use_container_width=True, hide_index=True)

with customers_tab:
    st.subheader("Top customers by lifetime value")
    customer_count = st.number_input(
        "Number of top customers",
        min_value=1,
        max_value=len(top_customers),
        value=min(20, len(top_customers)),
        step=1,
        help="Enter any number of customers to display, such as 20, 30, or 44.",
    )
    selected_customers = top_customers.head(int(customer_count))
    st.caption(f"Showing the top {len(selected_customers):,} of {len(top_customers):,} customers.")
    st.bar_chart(selected_customers.set_index("customer_id")["lifetime_value"])
    st.dataframe(selected_customers, use_container_width=True, hide_index=True)

with categories_tab:
    st.subheader("Sales by category")
    st.bar_chart(sales_by_category.set_index("category")[["revenue", "units_sold"]])
    st.dataframe(sales_by_category, use_container_width=True, hide_index=True)

with trends_tab:
    st.subheader("Monthly trends")
    chart_columns = st.multiselect(
        "Metrics",
        ["revenue", "order_count"],
        default=["revenue", "order_count"],
    )
    if chart_columns:
        st.line_chart(monthly_trends.set_index("month")[chart_columns])
    st.dataframe(monthly_trends, use_container_width=True, hide_index=True)

with performance_tab:
    st.subheader("Pandas vs PySpark performance")
    pandas_result = benchmark_pandas(orders, products)
    if st.button("Run PySpark benchmark", type="primary"):
        with st.spinner("Running the PySpark benchmark..."):
            pyspark_result = benchmark_pyspark_from_paths()
        st.session_state["pyspark_result"] = pyspark_result
    pyspark_result = st.session_state.get("pyspark_result")
    if pyspark_result is not None:
        comparison = pd.concat([pandas_result, pyspark_result], ignore_index=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("Engine")["Total (s)"])
    else:
        st.dataframe(pandas_result, use_container_width=True, hide_index=True)
        st.info("Click **Run PySpark benchmark** to add the PySpark timings.")
    st.caption("Timings vary by machine and are measured in the active Streamlit process.")
