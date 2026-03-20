"""Shared utilities for neuralviz: input handling, colors, backend selection."""

import warnings
import matplotlib.pyplot as plt

_CANONICAL_COLORS = [
    "#FF0000", "#0000FF", "#FFFF00", "#008000",
    "#FF00FF", "#00FFFF", "#808080", "#FFA500",
]

def normalize_input(code_or_obj):
    """Accept NeuralCode (duck typed), list, or tuple of binary strings."""
    if hasattr(code_or_obj, "Codes"):
        return list(code_or_obj.Codes)
    return list(code_or_obj)

def validate_code(code):
    """Validate that code is a non-empty list of equal-length binary strings."""
    if not code:
        raise ValueError("Neural code is empty — provide at least one codeword.")
    lengths = {len(c) for c in code}
    if len(lengths) > 1:
        raise ValueError(f"All codewords must have the same length, got lengths: {sorted(lengths)}")
    for c in code:
        if not all(ch in "01" for ch in c):
            raise ValueError(f"Codewords must be binary strings (0s and 1s only), got: '{c}'")
    n = len(code[0])
    if n > 8:
        warnings.warn(f"Code has {n} neurons — visualizations may be cluttered.", stacklevel=2)

def neuron_colors(n):
    """Return a list of n hex color strings, one per neuron."""
    colors = []
    for i in range(n):
        base_idx = i % 8
        cycle = i // 8
        base = _CANONICAL_COLORS[base_idx]
        if cycle == 0:
            colors.append(base)
        else:
            r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
            factor = 0.3 * cycle
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            colors.append(f"#{r:02X}{g:02X}{b:02X}")
    return colors

def get_backend(interactive):
    """Return 'plotly' or 'matplotlib' based on interactive flag."""
    if interactive:
        try:
            import plotly.graph_objects as go  # noqa: F401
            return "plotly"
        except ImportError:
            raise ImportError("plotly is required for interactive mode: pip install plotly")
    return "matplotlib"

def prepare_axes(ax, interactive=False):
    """Return (fig, ax) — create new figure if ax is None."""
    if interactive and ax is not None:
        raise ValueError("Cannot use ax parameter with interactive=True")
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.get_figure(), ax

def support(codeword):
    """Return the set of indices where the codeword is 1."""
    return frozenset(i for i, ch in enumerate(codeword) if ch == "1")
