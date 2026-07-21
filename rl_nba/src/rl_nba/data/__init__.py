"""Data sources: schema validation, synthetic generation, and loading."""

from rl_nba.data.loader import load_customers
from rl_nba.data.schema import (
    SchemaValidationError,
    ownership_matrix,
    required_columns,
    validate_customer_frame,
)
from rl_nba.data.synthetic import generate_customers

__all__ = [
    "SchemaValidationError",
    "generate_customers",
    "load_customers",
    "ownership_matrix",
    "required_columns",
    "validate_customer_frame",
]
