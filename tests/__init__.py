"""
Tests Package
=============
Unit tests for the transformer implementation.
"""

from .test_transformer import (
    test_transformer_equivalence,
    test_with_positional_encoding,
    test_without_positional_encoding,
    test_dimension_variations,
    generate_grading_outputs,
)

__all__ = [
    'test_transformer_equivalence',
    'test_with_positional_encoding',
    'test_without_positional_encoding',
    'test_dimension_variations',
    'generate_grading_outputs',
]
