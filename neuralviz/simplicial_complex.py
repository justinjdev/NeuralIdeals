"""Simplicial complex (nerve) visualization for neural codes."""

import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

from neuralviz._core import (
    normalize_input, validate_code, neuron_colors, get_backend, prepare_axes, support,
)


def _extract_simplices(code):
    """Return set of frozensets — each non-empty support is a simplex."""
    simplices = set()
    for codeword in code:
        s = support(codeword)
        if s:
            simplices.add(s)
    return simplices


def _maximal_simplices(simplices):
    """Return only the maximal simplices (not a face of any other)."""
    maximal = set()
    for s in simplices:
        is_face = any(s < other for other in simplices)
        if not is_face:
            maximal.add(s)
    return maximal


def _all_faces(simplices):
    """Expand simplices to include all faces (subsets)."""
    faces = set()
    for s in simplices:
        s_list = list(s)
        for i in range(1, 2 ** len(s_list)):
            face = frozenset(s_list[j] for j in range(len(s_list)) if i & (1 << j))
            faces.add(face)
    return faces


def _vertex_positions(vertices):
    """Place vertices evenly on a circle."""
    n = len(vertices)
    pos = {}
    for i, v in enumerate(vertices):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos[v] = (math.cos(angle), math.sin(angle))
    return pos


def plot_simplicial_complex(code, interactive=False, ax=None):
    """Draw the simplicial complex (nerve) of a neural code."""
    code = normalize_input(code)
    validate_code(code)
    backend = get_backend(interactive)

    raw_simplices = _extract_simplices(code)
    all_faces_set = _all_faces(raw_simplices)
    maximal = _maximal_simplices(raw_simplices)

    vertices = set()
    for s in all_faces_set:
        vertices |= s
    vertices = sorted(vertices)

    if backend == "plotly":
        return _plot_simplicial_plotly(vertices, all_faces_set, maximal)

    fig, ax_out = prepare_axes(ax, interactive=interactive)
    _plot_simplicial_matplotlib(vertices, all_faces_set, maximal, ax_out)
    if ax is not None:
        return ax_out
    return fig


def _plot_simplicial_matplotlib(vertices, all_faces, maximal, ax):
    pos = _vertex_positions(vertices)
    colors = neuron_colors(len(vertices))
    color_map = {v: colors[i] for i, v in enumerate(vertices)}

    for s in sorted(all_faces, key=len, reverse=True):
        if len(s) >= 3:
            pts = np.array([pos[v] for v in sorted(s)])
            cx, cy = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            order = np.argsort(angles)
            pts = pts[order]
            poly = MplPolygon(pts, alpha=0.15, facecolor="#8888FF", edgecolor="#6666CC", linewidth=1)
            ax.add_patch(poly)

    for s in all_faces:
        if len(s) == 2:
            v1, v2 = sorted(s)
            ax.plot([pos[v1][0], pos[v2][0]], [pos[v1][1], pos[v2][1]],
                    color="#666666", linewidth=2, zorder=1)

    for v in vertices:
        ax.plot(pos[v][0], pos[v][1], "o", color=color_map[v],
                markersize=15, zorder=2, markeredgecolor="#333", markeredgewidth=1.5)
        ax.annotate(f"x{v}", pos[v], textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    ax.set_aspect("equal")
    ax.set_title("Simplicial Complex", fontsize=12)
    ax.axis("off")


def _plot_simplicial_plotly(vertices, all_faces, maximal):
    import plotly.graph_objects as go

    pos = _vertex_positions(vertices)
    colors = neuron_colors(len(vertices))
    fig = go.Figure()

    for s in sorted(all_faces, key=len, reverse=True):
        if len(s) >= 3:
            verts = sorted(s)
            pts = np.array([pos[v] for v in verts])
            cx, cy = pts.mean(axis=0)
            angles_arr = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            order = np.argsort(angles_arr)
            pts = pts[order]
            fig.add_trace(go.Scatter(
                x=list(pts[:, 0]) + [pts[0, 0]], y=list(pts[:, 1]) + [pts[0, 1]],
                fill="toself", fillcolor="rgba(136,136,255,0.15)",
                line=dict(color="rgba(102,102,204,0.5)"),
                hoverinfo="skip", showlegend=False))

    for s in all_faces:
        if len(s) == 2:
            v1, v2 = sorted(s)
            fig.add_trace(go.Scatter(
                x=[pos[v1][0], pos[v2][0]], y=[pos[v1][1], pos[v2][1]],
                mode="lines", line=dict(width=2, color="#666"),
                hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Scatter(
        x=[pos[v][0] for v in vertices], y=[pos[v][1] for v in vertices],
        mode="markers+text",
        marker=dict(size=18, color=colors[:len(vertices)],
                    line=dict(width=1.5, color="#333")),
        text=[f"x{v}" for v in vertices], textposition="top center",
        hovertext=[f"Neuron {v}" for v in vertices], showlegend=False))

    fig.update_layout(title="Simplicial Complex", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig
