"""
General Utility Functions
=========================
General-purpose utility functions.
"""

from __future__ import annotations

import random

import numpy as np
import torch

# =============================================================================
# Type Aliases
# =============================================================================
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.floating]
TensorFloat = torch.Tensor

# =============================================================================
# Utility Functions
# =============================================================================

def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Backward compatibility alias
_set_seed = set_random_seed

