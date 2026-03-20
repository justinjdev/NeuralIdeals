"""Euler/Venn-style receptive field diagrams for neural codes."""

import math
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from neuralviz._core import (
    normalize_input, validate_code, neuron_colors, get_backend, prepare_axes, support,
)


def _extract_rf_relations(code):
    n_neurons = len(code[0])
    code_set = set(code)
    relations = {"containments": [], "disjoint": [], "covers": []}

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if not any(c[i] == "1" and c[j] == "1" for c in code):
                relations["disjoint"].append([i, j])

    for i in range(n_neurons):
        fires_i = [c for c in code if c[i] == "1"]
        if not fires_i:
            continue
        for j in range(n_neurons):
            if i == j:
                continue
            if all(c[j] == "1" for c in fires_i):
                relations["containments"].append(([i], [j]))

    non_zero = [c for c in code if c != "0" * n_neurons]
    if non_zero:
        found_covers = []
        for size in range(1, n_neurons + 1):
            for sigma in combinations(range(n_neurons), size):
                if all(any(c[s] == "1" for s in sigma) for c in non_zero):
                    is_minimal = not any(set(prev) < set(sigma) for prev in found_covers)
                    if is_minimal:
                        found_covers.append(list(sigma))
            if found_covers:
                break
        relations["covers"] = found_covers

    return relations


def _layout_circles(n_neurons, relations):
    base_radius = 1.0
    circles = []
    for i in range(n_neurons):
        angle = 2 * math.pi * i / n_neurons - math.pi / 2
        circles.append([2.0 * math.cos(angle), 2.0 * math.sin(angle), base_radius])

    for pair in relations.get("disjoint", []):
        i, j = pair[0], pair[1]
        dx = circles[j][0] - circles[i][0]
        dy = circles[j][1] - circles[i][1]
        dist = math.sqrt(dx * dx + dy * dy)
        min_dist = circles[i][2] + circles[j][2] + 0.3
        if dist < min_dist and dist > 0:
            scale = (min_dist - dist) / (2 * dist)
            circles[i][0] -= dx * scale
            circles[i][1] -= dy * scale
            circles[j][0] += dx * scale
            circles[j][1] += dy * scale

    for sigma, tau in relations.get("containments", []):
        i, j = sigma[0], tau[0]
        circles[i][2] = circles[j][2] * 0.6
        circles[i][0] = circles[j][0] + circles[j][2] * 0.2
        circles[i][1] = circles[j][1] + circles[j][2] * 0.2

    return [(c[0], c[1], c[2]) for c in circles]


def plot_receptive_fields(code, rf_structure=None, interactive=False, ax=None):
    code = normalize_input(code)
    validate_code(code)
    backend = get_backend(interactive)

    n_neurons = len(code[0])
    relations = rf_structure if rf_structure is not None else _extract_rf_relations(code)
    circles = _layout_circles(n_neurons, relations)

    if backend == "plotly":
        return _plot_rf_plotly(n_neurons, circles, relations)

    fig, ax_out = prepare_axes(ax, interactive=interactive)
    _plot_rf_matplotlib(n_neurons, circles, relations, ax_out)
    if ax is not None:
        return ax_out
    return fig


def _plot_rf_matplotlib(n_neurons, circles, relations, ax):
    colors = neuron_colors(n_neurons)
    for i, (cx, cy, r) in enumerate(circles):
        ellipse = Ellipse((cx, cy), width=r * 2, height=r * 2,
                          facecolor=colors[i], alpha=0.2, edgecolor=colors[i], linewidth=2)
        ax.add_patch(ellipse)
        ax.text(cx, cy, f"U{i}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=colors[i])

    if circles:
        all_x = [c[0] for c in circles]
        all_r = [c[2] for c in circles]
        all_y = [c[1] for c in circles]
        margin = 1.0
        ax.set_xlim(min(x - r for x, r in zip(all_x, all_r)) - margin,
                    max(x + r for x, r in zip(all_x, all_r)) + margin)
        ax.set_ylim(min(y - r for y, r in zip(all_y, all_r)) - margin,
                    max(y + r for y, r in zip(all_y, all_r)) + margin)

    ax.set_aspect("equal")
    ax.set_title("Receptive Field Structure", fontsize=12)
    ax.axis("off")


def _plot_rf_plotly(n_neurons, circles, relations):
    import plotly.graph_objects as go

    colors = neuron_colors(n_neurons)
    fig = go.Figure()
    for i, (cx, cy, r) in enumerate(circles):
        fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                      fillcolor=colors[i], opacity=0.2, line=dict(color=colors[i], width=2))
        fig.add_annotation(x=cx, y=cy, text=f"U{i}", showarrow=False,
                           font=dict(size=14, color=colors[i]))

    fig.update_layout(title="Receptive Field Structure", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig
