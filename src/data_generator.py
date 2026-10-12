"""Generate synthetic e-commerce customers, products, and orders."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import Settings, configure_logging, create_data_directories

LOGGER = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Create reproducible e-commerce datasets as pandas DataFrames.

    Twenty percent of customers receive 80 percent of orders. NumPy Pareto
    weights determine how orders are shared within each customer segment.
    """

    CATEGORIES: tuple[str, ...] = ("Electronics", "Clothing", "Home", "Sports", "Books")

    def __init__(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        """Initialize output sizes and seeded random generators.

        Args:
            customer_count: Number of customers to generate.
            product_count: Number of products to generate.
            order_count: Number of orders to generate.
            seed: Seed used for Faker and NumPy random generation.
        """
        self._validate_counts(customer_count, product_count, order_count)
        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)

    @staticmethod
    def _validate_counts(customer_count: int, product_count: int, order_count: int) -> None:
        """Ensure requested dataset sizes are positive.

        Raises:
            ValueError: If any requested count is less than one.
        """
        if any(count < 1 for count in (customer_count, product_count, order_count)):
            raise ValueError("Customer, product, and order counts must all be at least 1.")

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer identity, location, age, and registration data.

        Ages follow a normal distribution centered on 35 and are clipped to
        the inclusive range from 18 to 90.

        Returns:
            A DataFrame with one row per generated customer.
        """
        LOGGER.info("Generating %s customers.", f"{self.customer_count:,}")
        today = date.today()
        registration_offsets = self.rng.integers(0, 1_096, size=self.customer_count)
        registrations = [today - timedelta(days=int(offset)) for offset in registration_offsets]
        names: list[str] = []
        emails: list[str] = []
        cities: list[str] = []
        countries: list[str] = []

        for _ in tqdm(range(self.customer_count), desc="Generating customers", unit="customer"):
            names.append(self.fake.name())
            emails.append(self.fake.unique.email())
            cities.append(self.fake.city())
            countries.append(self.fake.country())

        return pd.DataFrame(
            {
                "customer_id": np.arange(1, self.customer_count + 1, dtype=np.int64),
                "name": names,
                "email": emails,
                "age": np.clip(self.rng.normal(35, 12, self.customer_count).round(), 18, 90).astype(int),
                "city": cities,
                "country": countries,
                "registration_date": pd.to_datetime(registrations),
            }
        )

    def generate_products(self) -> pd.DataFrame:
        """Generate products with category, price, stock, and rating data.

        Returns:
            A DataFrame with one row per generated product.
        """
        LOGGER.info("Generating %s products.", f"{self.product_count:,}")
        names = [
            f"{self.fake.word().title()} {self.fake.word().title()}"
            for _ in tqdm(range(self.product_count), desc="Generating products", unit="product")
        ]
        return pd.DataFrame(
            {
                "product_id": np.arange(1, self.product_count + 1, dtype=np.int64),
                "name": names,
                "category": self.rng.choice(self.CATEGORIES, size=self.product_count),
                "price": self.rng.uniform(10, 500, size=self.product_count).round(2),
                "stock": self.rng.integers(0, 1_001, size=self.product_count),
                "rating": self.rng.uniform(1, 5, size=self.product_count).round(1),
            }
        )

    def _pareto_probabilities(self, size: int) -> np.ndarray:
        """Create normalized Pareto-distributed sampling probabilities.

        Args:
            size: Number of customers in the sampled segment.

        Returns:
            Probabilities that sum to one.
        """
        weights = self.rng.pareto(a=1.5, size=size) + 1
        return weights / weights.sum()

    def _customer_order_ids(self, customer_ids: np.ndarray) -> np.ndarray:
        """Allocate orders using a Pareto-weighted 80/20 customer split.

        Args:
            customer_ids: IDs eligible to receive generated orders.

        Returns:
            A customer ID for every generated order.
        """
        high_value_count = max(1, int(len(customer_ids) * 0.2))
        high_value_orders = int(self.order_count * 0.8)
        remaining_customers = len(customer_ids) - high_value_count
        remaining_orders = self.order_count - high_value_orders
        high_value_ids = self.rng.choice(
            customer_ids[:high_value_count],
            size=high_value_orders,
            p=self._pareto_probabilities(high_value_count),
        )
        if remaining_customers == 0 or remaining_orders == 0:
            return high_value_ids
        other_ids = self.rng.choice(
            customer_ids[high_value_count:],
            size=remaining_orders,
            p=self._pareto_probabilities(remaining_customers),
        )
        customer_ids = np.concatenate((high_value_ids, other_ids))
        self.rng.shuffle(customer_ids)
        return customer_ids

    def generate_orders(self, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """Generate orders linked to supplied customer and product records.

        Args:
            customers: Customer DataFrame with a ``customer_id`` column.
            products: Product DataFrame with a ``product_id`` column.

        Returns:
            An orders DataFrame with quantity values from 1 to 10.

        Raises:
            ValueError: If input data is empty or lacks required ID columns.
        """
        if customers.empty or products.empty:
            raise ValueError("Customers and products must contain at least one row.")
        if "customer_id" not in customers or "product_id" not in products:
            raise ValueError("Customers need customer_id and products need product_id columns.")

        LOGGER.info("Generating %s orders using a Pareto 80/20 allocation.", f"{self.order_count:,}")
        start_date = np.datetime64(date.today() - timedelta(days=365))
        day_offsets = self.rng.integers(0, 366, size=self.order_count).astype("timedelta64[D]")
        return pd.DataFrame(
            {
                "order_id": np.arange(1, self.order_count + 1, dtype=np.int64),
                "customer_id": self._customer_order_ids(customers["customer_id"].to_numpy()),
                "product_id": self.rng.choice(products["product_id"].to_numpy(), size=self.order_count),
                "quantity": self.rng.integers(1, 11, size=self.order_count),
                "order_date": pd.to_datetime(start_date + day_offsets),
            }
        )

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the complete customer, product, and order datasets.

        Returns:
            Customers, products, and orders DataFrames in that order.
        """
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers, products)
        return customers, products, orders

    @staticmethod
    def save_dataframes(
        customers: pd.DataFrame,
        products: pd.DataFrame,
        orders: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Save generated DataFrames as UTF-8 CSV files.

        Args:
            customers: Customer data to save.
            products: Product data to save.
            orders: Order data to save.
            output_dir: Destination directory for CSV files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, dataframe in (("customers", customers), ("products", products), ("orders", orders)):
            destination = output_dir / f"{name}.csv"
            dataframe.to_csv(destination, index=False)
            LOGGER.info("Wrote %s records to %s", f"{len(dataframe):,}", destination)


def parse_arguments() -> argparse.Namespace:
    """Parse optional record-count and seed arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce data.")
    parser.add_argument("--customers", type=int, default=100_000, help="Number of customers to generate.")
    parser.add_argument("--products", type=int, default=10_000, help="Number of products to generate.")
    parser.add_argument("--orders", type=int, default=1_000_000, help="Number of orders to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data.")
    return parser.parse_args()


def main() -> None:
    """Generate data with the requested sizes and save it to ``data/raw``."""
    arguments = parse_arguments()
    configure_logging()
    settings = Settings()
    create_data_directories(settings)
    generator = SyntheticDataGenerator(arguments.customers, arguments.products, arguments.orders, arguments.seed)
    customers, products, orders = generator.generate_all()
    generator.save_dataframes(customers, products, orders, settings.raw_data_dir)


if __name__ == "__main__":
    main()
