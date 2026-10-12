"""Tests for the synthetic e-commerce data generator."""

from src.data_generator import SyntheticDataGenerator


def test_generated_orders_reference_available_records() -> None:
    """Generated orders use supplied customer and product IDs."""
    generator = SyntheticDataGenerator(customer_count=10, product_count=3, order_count=50, seed=42)
    customers, products, orders = generator.generate_all()

    customer_ids = set(customers["customer_id"])
    product_ids = set(products["product_id"])
    assert len(orders) == 50
    assert set(orders["customer_id"]).issubset(customer_ids)
    assert set(orders["product_id"]).issubset(product_ids)
    assert orders["quantity"].between(1, 10).all()
