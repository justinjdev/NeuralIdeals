"""neuralviz — Pure Python visualization package for neural codes."""

__version__ = "0.1.0"

from neuralviz.hasse import plot_hasse
from neuralviz.simplicial_complex import plot_simplicial_complex
from neuralviz.convex_realization import plot_convex_1d, plot_convex_2d
from neuralviz.receptive_fields import plot_receptive_fields
from neuralviz.legacy import plot_arcs, plot_lines, plot_circles

__all__ = [
    "plot_hasse",
    "plot_simplicial_complex",
    "plot_convex_1d",
    "plot_convex_2d",
    "plot_receptive_fields",
    "plot_arcs",
    "plot_lines",
    "plot_circles",
]
