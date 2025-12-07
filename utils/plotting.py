"""
Plotting Functions
==================
Functions for visualization and saving plots.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .constants import PLOT_CONFIG
from .utils import NDArrayFloat


def rescale_and_plot(
    arr: NDArrayFloat,
    title: str = '',
    ax: Optional[Axes] = None,
    x_lab: Optional[str] = None,
    y_lab: Optional[str] = None,
    colormap: str = PLOT_CONFIG.colormap,
    vmin: Optional[float] = PLOT_CONFIG.vmin,
    vmax: Optional[float] = PLOT_CONFIG.vmax,
    title_fontsize: int = PLOT_CONFIG.title_fontsize,
    label_fontsize: int = PLOT_CONFIG.label_fontsize,
    show_xticks: bool = PLOT_CONFIG.show_xticks,
    show_yticks: bool = PLOT_CONFIG.show_yticks,
) -> None:
    """Plot a matrix as a heatmap with automatic [0, 1] rescaling."""
    assert arr.ndim == 2, f"arr must be 2D, got shape {arr.shape}"
    
    # Rescale to [0, 1]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    
    ax.imshow(arr, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=title_fontsize)
    
    if not show_xticks:
        ax.set_xticks([])
    if not show_yticks:
        ax.set_yticks([])
    if x_lab is not None:
        ax.set_xlabel(x_lab, fontsize=label_fontsize)
    if y_lab is not None:
        ax.set_ylabel(y_lab, fontsize=label_fontsize)


def save_figure(
    fig: Figure,
    name: str,
    save_dir: str = PLOT_CONFIG.save_dir,
    save_format: str = PLOT_CONFIG.save_format,
    save_dpi: int = PLOT_CONFIG.save_dpi,
    add_timestamp: bool = PLOT_CONFIG.add_timestamp,
    run_name: Optional[str] = None,
) -> Path:
    """Save a matplotlib figure to the plots directory.
    
    Args:
        fig: Matplotlib figure to save.
        name: Base name for the file (without extension).
        save_dir: Directory to save plots (created if doesn't exist).
        save_format: File format ("png", "pdf", "svg").
        save_dpi: Resolution for rasterized formats.
        add_timestamp: If True, append timestamp to filename.
        run_name: Optional run identifier to organize plots by experiment.
    
    Returns:
        Path to the saved figure.
    """
    # Create directory structure
    save_path = Path(save_dir)
    if run_name:
        save_path = save_path / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{save_format}"
    else:
        filename = f"{name}.{save_format}"
    
    filepath = save_path / filename
    fig.savefig(filepath, dpi=save_dpi, bbox_inches='tight', format=save_format)
    print(f"✓ Figure saved: {filepath}")
    return filepath


def get_current_figure() -> Figure:
    """Get the current matplotlib figure."""
    return plt.gcf()

