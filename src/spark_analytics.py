"""Analyze generated e-commerce data with PySpark."""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import Settings, configure_logging, create_data_directories

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Provide named analytics operations over the raw sales DataFrames."""

    def __init__(self, customers: DataFrame, products: DataFrame, orders: DataFrame) -> None:
        self.customers = customers
        self.products = products
        self.orders = orders_with_revenue(orders, products)

    def top_customers(self) -> DataFrame:
        """Return customers ranked by lifetime value."""
        return customer_value(self.orders, self.customers)

    def sales_by_category(self) -> DataFrame:
        """Return sales totals grouped by product category."""
        return category_sales(self.orders, self.products)

    def monthly_trends(self) -> DataFrame:
        """Return revenue and order counts grouped by month."""
        return monthly_revenue(self.orders)


def create_spark_session() -> SparkSession:
    """Create a local Spark session for the analytics job.

    Returns:
        An active local Spark session.
    """
    return (
        SparkSession.builder.appName("EcommerceAnalytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def read_raw_data(spark: SparkSession, raw_data_dir: Path) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Load customers, products, and orders CSV files into Spark DataFrames.

    Args:
        spark: Spark session used for reading files.
        raw_data_dir: Directory containing the generated CSV files.

    Returns:
        Customers, products, and orders DataFrames in that order.
    """
    reader = spark.read.option("header", True).option("inferSchema", True)
    return (
        reader.csv(str(raw_data_dir / "customers.csv")),
        reader.csv(str(raw_data_dir / "products.csv")),
        reader.csv(str(raw_data_dir / "orders.csv")),
    )


def orders_with_revenue(orders: DataFrame, products: DataFrame) -> DataFrame:
    """Join orders to their product details and calculate revenue.

    Args:
        orders: Source orders DataFrame.
        products: Product catalog DataFrame containing category and price.

    Returns:
        Orders enriched with product category and total amount.
    """
    return orders.join(products.select("product_id", "category", "price"), "product_id").withColumn(
        "total_amount", F.round(F.col("quantity") * F.col("price"), 2)
    )


def category_sales(orders: DataFrame, products: DataFrame) -> DataFrame:
    """Calculate order revenue and units by product category.

    Args:
        orders: Revenue-enriched orders DataFrame.
        products: Product catalog DataFrame.

    Returns:
        Category summary sorted by revenue descending.
    """
    return (
        orders.groupBy("category")
        .agg(
            F.round(F.sum("total_amount"), 2).alias("revenue"),
            F.sum("quantity").alias("units_sold"),
            F.count("order_id").alias("order_count"),
        )
        .orderBy(F.desc("revenue"))
    )


def customer_value(orders: DataFrame, customers: DataFrame) -> DataFrame:
    """Calculate spend and order count for each customer.

    Args:
        orders: Revenue-enriched orders DataFrame.
        customers: Customer DataFrame.

    Returns:
        Customer value summary sorted by lifetime spend descending.
    """
    return (
        orders.groupBy("customer_id")
        .agg(
            F.round(F.sum("total_amount"), 2).alias("lifetime_value"),
            F.count("order_id").alias("order_count"),
        )
        .join(customers.select("customer_id", "name", "city", "country"), "customer_id")
        .orderBy(F.desc("lifetime_value"))
    )


def monthly_revenue(orders: DataFrame) -> DataFrame:
    """Calculate order revenue for each calendar month.

    Args:
        orders: Revenue-enriched orders DataFrame.

    Returns:
        Monthly revenue and order-count summary in chronological order.
    """
    return (
        orders.withColumn("month", F.date_format(F.to_date("order_date"), "yyyy-MM"))
        .groupBy("month")
        .agg(F.round(F.sum("total_amount"), 2).alias("revenue"), F.count("order_id").alias("order_count"))
        .orderBy("month")
    )


def write_result(dataframe: DataFrame, destination: Path) -> None:
    """Write one analytics DataFrame as a single CSV file.

    Args:
        dataframe: Result DataFrame to persist.
        destination: Directory for the CSV result.
    """
    dataframe.coalesce(1).write.mode("overwrite").option("header", True).csv(str(destination))
    LOGGER.info("Wrote analysis result to %s", destination)


def run_analytics(settings: Settings) -> None:
    """Read raw data, calculate insights, and save the results.

    Args:
        settings: Project paths used for input and output.
    """
    create_data_directories(settings)
    spark = create_spark_session()
    try:
        customers, products, orders = read_raw_data(spark, settings.raw_data_dir)
        enriched_orders = orders_with_revenue(orders, products)
        write_result(category_sales(enriched_orders, products), settings.processed_data_dir / "category_sales")
        write_result(customer_value(enriched_orders, customers), settings.processed_data_dir / "customer_value")
        write_result(monthly_revenue(enriched_orders), settings.processed_data_dir / "monthly_revenue")
    finally:
        spark.stop()


def main() -> None:
    """Run the Spark analytics job from the command line."""
    configure_logging()
    run_analytics(Settings())


if __name__ == "__main__":
    main()
