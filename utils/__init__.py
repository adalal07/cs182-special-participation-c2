"""
Utils Package
=============
Utility modules for the transformer notebook.
"""

from .constants import (
    PlotConfig,
    TrainingConfig,
    PLOT_CONFIG,
    TRAIN_CONFIG,
    RELATIVE_TOLERANCE,
    TEST_ITERATIONS,
    TEST_MIN_SEQ_LEN,
    TEST_MAX_SEQ_LEN,
    TEST_INPUT_DIM,
)

from .utils import (
    set_random_seed,
    _set_seed,
    NDArrayFloat,
    TensorFloat,
)

from .plotting import (
    rescale_and_plot,
    save_figure,
    get_current_figure,
)

from .training import (
    train_loop,
)

from .comparison import (
    compare_transformers,
)

__all__ = [
    # Config classes
    'PlotConfig',
    'TrainingConfig',
    'PLOT_CONFIG',
    'TRAIN_CONFIG',
    # Constants
    'RELATIVE_TOLERANCE',
    'TEST_ITERATIONS',
    'TEST_MIN_SEQ_LEN',
    'TEST_MAX_SEQ_LEN',
    'TEST_INPUT_DIM',
    # Utility functions
    'set_random_seed',
    '_set_seed',
    # Plotting functions
    'rescale_and_plot',
    'save_figure',
    'get_current_figure',
    # Training functions
    'train_loop',
    # Comparison functions
    'compare_transformers',
    # Type aliases
    'NDArrayFloat',
    'TensorFloat',
]
