"""Convex realization visualizations — intervals (R¹) and polygons (R²)."""

import math
import warnings
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from neuralviz._core import (
    normalize_input, validate_code, neuron_colors, get_backend, prepare_axes, support,
)


def _build_ordering_constraints(code):
    n_neurons = len(code[0])
    code_set = sorted(set(code))
    constraints = []
    for neuron in range(n_neurons):
        firing = [c for c in code_set if c[neuron] == "1"]
        if firing:
            constraints.append((neuron, firing))
    return constraints


def _find_valid_ordering(code):
    """Find ordering of codewords where each neuron's firing set is contiguous."""
    code_set = sorted(set(code))
    n = len(code_set)
    n_neurons = len(code[0])

    def is_contiguous(ordering):
        for neuron in range(n_neurons):
            firing_indices = [i for i, c in enumerate(ordering) if c[neuron] == "1"]
            if firing_indices:
                if firing_indices[-1] - firing_indices[0] != len(firing_indices) - 1:
                    return False
        return True

    if n <= 8:
        for perm in permutations(code_set):
            if is_contiguous(perm):
                return list(perm)
        return None

    for start in code_set:
        ordering = [start]
        remaining = set(code_set) - {start}
        while remaining:
            best = None
            for candidate in remaining:
                test = ordering + [candidate]
                valid = True
                for neuron in range(n_neurons):
                    firing = [i for i, c in enumerate(test) if c[neuron] == "1"]
                    non_firing_between = any(
                        test[k][neuron] == "0"
                        for k in range(min(firing), max(firing) + 1)
                    ) if len(firing) > 1 else False
                    if non_firing_between:
                        valid = False
                        break
                if valid:
                    best = candidate
                    break
            if best is None:
                break
            ordering.append(best)
            remaining.remove(best)
        if not remaining and is_contiguous(ordering):
            return ordering
    return None


def _compute_intervals(code):
    n_neurons = len(code[0])
    ordering = _find_valid_ordering(code)
    if ordering is None:
        ordering = sorted(set(code))

    point_map = {c: i for i, c in enumerate(ordering)}
    intervals = []
    for neuron in range(n_neurons):
        firing_points = [point_map[c] for c in ordering if c[neuron] == "1"]
        if not firing_points:
            intervals.append((-(neuron + 1) * 0.5 - 0.3, -(neuron + 1) * 0.5 - 0.1))
        else:
            left = min(firing_points) - 0.4
            right = max(firing_points) + 0.4
            intervals.append((left, right))
    return intervals


def _verify_realization(code, intervals):
    code_set = set(code)
    code_set.add("0" * len(code[0]))  # all-zeros codeword is always implicit
    n_neurons = len(code[0])
    ordering = _find_valid_ordering(code)
    if ordering is None:
        ordering = sorted(set(code))
    n_points = len(ordering)

    for idx, c in enumerate(ordering):
        expected = support(c)
        actual = frozenset(i for i in range(n_neurons) if intervals[i][0] <= idx <= intervals[i][1])
        if expected != actual:
            return False

    for idx in range(n_points - 1):
        midpt = idx + 0.5
        mid_support = frozenset(i for i in range(n_neurons) if intervals[i][0] <= midpt <= intervals[i][1])
        mid_code = "".join("1" if i in mid_support else "0" for i in range(n_neurons))
        if mid_code not in code_set:
            return False

    for test_pt in [-1.0, n_points + 0.5]:
        outer_support = frozenset(i for i in range(n_neurons) if intervals[i][0] <= test_pt <= intervals[i][1])
        outer_code = "".join("1" if i in outer_support else "0" for i in range(n_neurons))
        if outer_code not in code_set:
            return False

    return True


def plot_convex_1d(code, interactive=False, ax=None, best_effort=False):
    code = normalize_input(code)
    validate_code(code)
    backend = get_backend(interactive)

    intervals = _compute_intervals(code)
    if not _verify_realization(code, intervals):
        if best_effort:
            warnings.warn("Code may not have a valid R¹ convex realization — showing best-effort layout.", stacklevel=2)
        else:
            raise ValueError("Code does not have a valid convex realization in R¹. Pass best_effort=True to see a best-effort layout.")

    if backend == "plotly":
        return _plot_convex_1d_plotly(code, intervals)

    fig, ax_out = prepare_axes(ax, interactive=interactive)
    _plot_convex_1d_matplotlib(code, intervals, ax_out)
    if ax is not None:
        return ax_out
    return fig


def _plot_convex_1d_matplotlib(code, intervals, ax):
    n_neurons = len(intervals)
    colors = neuron_colors(n_neurons)
    height = 0.6
    spacing = 1.0

    for i, (left, right) in enumerate(intervals):
        y = i * spacing
        width = right - left
        rect = FancyBboxPatch((left, y - height / 2), width, height,
                              boxstyle="round,pad=0.05", facecolor=colors[i],
                              alpha=0.4, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        ax.text(left - 0.2, y, f"U{i}", ha="right", va="center",
                fontsize=10, fontweight="bold", color=colors[i])

    sorted_code = sorted(set(code))
    for j, c in enumerate(sorted_code):
        ax.axvline(x=j, color="#999999", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(j, -0.8, c, ha="center", va="top", fontsize=8, family="monospace")

    ax.set_xlim(min(iv[0] for iv in intervals) - 1, max(iv[1] for iv in intervals) + 1)
    ax.set_ylim(-1.5, n_neurons * spacing)
    ax.set_title("Convex Realization (R¹)", fontsize=12)
    ax.set_yticks([])
    ax.set_xlabel("Stimulus space")


def _plot_convex_1d_plotly(code, intervals):
    import plotly.graph_objects as go

    n_neurons = len(intervals)
    colors = neuron_colors(n_neurons)
    fig = go.Figure()

    for i, (left, right) in enumerate(intervals):
        fig.add_shape(type="rect", x0=left, x1=right, y0=i - 0.3, y1=i + 0.3,
                      fillcolor=colors[i], opacity=0.4, line=dict(color=colors[i], width=2))
        fig.add_annotation(x=left - 0.3, y=i, text=f"U{i}", showarrow=False,
                           font=dict(size=12, color=colors[i]))

    sorted_code = sorted(set(code))
    for j, c in enumerate(sorted_code):
        fig.add_vline(x=j, line_dash="dot", line_color="#999", opacity=0.5)
        fig.add_annotation(x=j, y=-0.6, text=c, showarrow=False,
                           font=dict(size=10, family="monospace"))

    fig.update_layout(title="Convex Realization (R¹)", showlegend=False,
                      xaxis_title="Stimulus space", yaxis=dict(showticklabels=False))
    return fig
