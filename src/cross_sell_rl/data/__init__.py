"""Data sources: schema validation, synthetic generation, and loading."""

from cross_sell_rl.data.loader import load_customers
from cross_sell_rl.data.schema import (
    SchemaValidationError,
    ownership_matrix,
    required_columns,
    validate_customer_frame,
)
from cross_sell_rl.data.synthetic import generate_customers

__all__ = [
    "SchemaValidationError",
    "generate_customers",
    "load_customers",
    "ownership_matrix",
    "required_columns",
    "validate_customer_frame",
]
