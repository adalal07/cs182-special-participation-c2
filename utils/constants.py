"""
Constants and Configuration Classes
====================================
This module contains configuration classes and constants used throughout the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt

# =============================================================================
# Type Aliases
# =============================================================================
# Note: These are re-exported here for convenience, but actual type aliases
# should be defined where numpy.typing is imported

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for matplotlib plots."""
    # Figure settings
    figure_width: float = 20.0
    figure_height: float = 5.0
    dpi: int = 100
    
    # Colormap settings
    colormap: str = "Reds"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    
    # Text settings
    title_fontsize: int = 12
    label_fontsize: int = 10
    
    # Axis settings
    show_xticks: bool = False
    show_yticks: bool = False
    tight_layout: bool = True
    
    # Grid settings
    nrows: int = 1
    ncols: int = 8
    
    # Save settings
    save_dir: str = "plots"
    save_format: str = "png"  # "png", "pdf", "svg"
    save_dpi: int = 150
    auto_save: bool = False  # If True, automatically save all plots
    add_timestamp: bool = True  # Add timestamp to filenames


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    num_epochs: int = 10_001
    learning_rate: float = 3e-2
    log_interval: int = 1000
    optimizer: str = "sgd"  # "sgd" or "adam"
    momentum: float = 0.0
    weight_decay: float = 0.0
    loss_fn: str = "mse"  # "mse" or "cross_entropy"


# =============================================================================
# Default Configurations
# =============================================================================
PLOT_CONFIG = PlotConfig()
TRAIN_CONFIG = TrainingConfig()

# Initialize matplotlib with default plot config
plt.rcParams['figure.figsize'] = [PLOT_CONFIG.figure_width, PLOT_CONFIG.figure_height]
plt.rcParams['figure.dpi'] = PLOT_CONFIG.dpi

# =============================================================================
# Constants
# =============================================================================
RELATIVE_TOLERANCE = 1e-3
TEST_ITERATIONS = 10
TEST_MIN_SEQ_LEN = 1
TEST_MAX_SEQ_LEN = 4
TEST_INPUT_DIM = 5

