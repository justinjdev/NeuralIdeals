"""Python ports of the Java arc/line/circle visualization engines."""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch

from neuralviz._core import (
    normalize_input, validate_code, neuron_colors, get_backend, prepare_axes,
)


def plot_lines(code, interactive=False, ax=None):
    code = normalize_input(code)
    validate_code(code)
    backend = get_backend(interactive)
    if backend == "plotly":
        return _plot_lines_plotly(code)
    fig, ax_out = prepare_axes(ax, interactive=interactive)
    _plot_lines_matplotlib(code, ax_out)
    if ax is not None:
        return ax_out
    return fig


def _plot_lines_matplotlib(code, ax):
    n_neurons = len(code[0])
    n_codes = len(code)
    colors = neuron_colors(n_neurons)
    cell_w, cell_h = 1.0, 0.6
    gap = 0.1

    for col, codeword in enumerate(code):
        for row in range(n_neurons):
            x = col * (cell_w + gap)
            y = (n_neurons - 1 - row) * (cell_h + gap)
            if codeword[row] == "1":
                rect = FancyBboxPatch((x, y), cell_w, cell_h, boxstyle="round,pad=0.02",
                                     facecolor=colors[row], alpha=0.7, edgecolor="#333333", linewidth=1)
                ax.add_patch(rect)

    for row in range(n_neurons):
        y = (n_neurons - 1 - row) * (cell_h + gap) + cell_h / 2
        ax.text(-0.5, y, f"N{row}", ha="right", va="center", fontsize=9, fontweight="bold", color=colors[row])

    for col, codeword in enumerate(code):
        x = col * (cell_w + gap) + cell_w / 2
        ax.text(x, -0.4, codeword, ha="center", va="top", fontsize=8, family="monospace")

    ax.set_xlim(-1, n_codes * (cell_w + gap))
    ax.set_ylim(-0.8, n_neurons * (cell_h + gap))
    ax.set_aspect("equal")
    ax.set_title("Line Display", fontsize=12)
    ax.axis("off")


def _plot_lines_plotly(code):
    import plotly.graph_objects as go
    n_neurons = len(code[0])
    colors = neuron_colors(n_neurons)
    fig = go.Figure()
    cell_w, cell_h, gap = 1.0, 0.6, 0.1

    for col, codeword in enumerate(code):
        for row in range(n_neurons):
            if codeword[row] == "1":
                x = col * (cell_w + gap)
                y = (n_neurons - 1 - row) * (cell_h + gap)
                fig.add_shape(type="rect", x0=x, y0=y, x1=x + cell_w, y1=y + cell_h,
                              fillcolor=colors[row], opacity=0.7, line=dict(color="#333", width=1))

    for row in range(n_neurons):
        y = (n_neurons - 1 - row) * (cell_h + gap) + cell_h / 2
        fig.add_annotation(x=-0.3, y=y, text=f"N{row}", showarrow=False, font=dict(size=10, color=colors[row]))

    for col, codeword in enumerate(code):
        x = col * (cell_w + gap) + cell_w / 2
        fig.add_annotation(x=x, y=-0.3, text=codeword, showarrow=False, font=dict(size=9, family="monospace"))

    fig.update_layout(title="Line Display", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig


def plot_arcs(code, interactive=False, ax=None):
    code = normalize_input(code)
    validate_code(code)
    backend = get_backend(interactive)
    if backend == "plotly":
        return _plot_arcs_plotly(code)
    fig, ax_out = prepare_axes(ax, interactive=interactive)
    _plot_arcs_matplotlib(code, ax_out)
    if ax is not None:
        return ax_out
    return fig


def _plot_arcs_matplotlib(code, ax):
    n_neurons = len(code[0])
    n_codes = len(code)
    colors = neuron_colors(n_neurons)
    base_radius = 1.0
    radius_step = 0.3
    arc_span = 360.0 / n_codes

    for neuron_idx in range(n_neurons):
        r = base_radius + neuron_idx * radius_step
        color = colors[neuron_idx]
        for code_idx in range(n_codes):
            if code[code_idx][neuron_idx] == "1":
                start_angle = code_idx * arc_span
                arc = Arc((0, 0), 2 * r, 2 * r, angle=0,
                          theta1=start_angle, theta2=start_angle + arc_span,
                          color=color, linewidth=3)
                ax.add_patch(arc)

    inner = plt.Circle((0, 0), base_radius * 0.8, color="#CCCCCC", alpha=0.3)
    ax.add_patch(inner)

    for i in range(n_neurons):
        r_label = base_radius + n_neurons * radius_step + 0.3
        angle = math.pi / 2 + i * 0.4
        ax.text(r_label * math.cos(angle), r_label * math.sin(angle),
                f"N{i}", color=colors[i], fontsize=9, fontweight="bold", ha="center", va="center")

    max_r = base_radius + n_neurons * radius_step + 0.8
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect("equal")
    ax.set_title("Arc Display", fontsize=12)
    ax.axis("off")


def _plot_arcs_plotly(code):
    import plotly.graph_objects as go
    n_neurons = len(code[0])
    n_codes = len(code)
    colors = neuron_colors(n_neurons)
    base_radius = 1.0
    radius_step = 0.3
    arc_span = 360.0 / n_codes
    fig = go.Figure()

    for neuron_idx in range(n_neurons):
        r = base_radius + neuron_idx * radius_step
        color = colors[neuron_idx]
        for code_idx in range(n_codes):
            if code[code_idx][neuron_idx] == "1":
                start = math.radians(code_idx * arc_span)
                end = math.radians((code_idx + 1) * arc_span)
                theta = np.linspace(start, end, 30)
                fig.add_trace(go.Scatter(
                    x=(r * np.cos(theta)).tolist(), y=(r * np.sin(theta)).tolist(),
                    mode="lines", line=dict(width=4, color=color),
                    hoverinfo="text", hovertext=f"Neuron {neuron_idx}", showlegend=False))

    fig.update_layout(title="Arc Display", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig


def plot_circles(code, interactive=False, ax=None):
    raise NotImplementedError(
        "plot_circles is not yet implemented — the source for "
        "InductiveCircles.jar is missing. Provide the Java source or "
        "decompile the JAR to enable this visualization."
    )
