"""Hasse diagram of the codeword poset ordered by support containment."""

import networkx as nx
import matplotlib.pyplot as plt

from neuralviz._core import (
    normalize_input, validate_code, neuron_colors, get_backend, prepare_axes, support,
)


def _build_poset(code):
    """Return set of (c1, c2) pairs where supp(c1) ⊆ supp(c2) and c1 != c2."""
    edges = set()
    for c1 in code:
        s1 = support(c1)
        for c2 in code:
            if c1 != c2:
                s2 = support(c2)
                if s1 <= s2:
                    edges.add((c1, c2))
    return edges


def _covering_relations(code, all_edges):
    """Remove transitive edges to get covering relations."""
    covers = set()
    for c1, c2 in all_edges:
        is_cover = True
        for c3 in code:
            if c3 != c1 and c3 != c2:
                if (c1, c3) in all_edges and (c3, c2) in all_edges:
                    is_cover = False
                    break
        if is_cover:
            covers.add((c1, c2))
    return covers


def _hamming_weight(codeword):
    return sum(1 for ch in codeword if ch == "1")


def plot_hasse(code, interactive=False, ax=None):
    """Draw a Hasse diagram of the codeword poset."""
    code = normalize_input(code)
    validate_code(code)
    backend = get_backend(interactive)

    all_edges = _build_poset(code)
    covers = _covering_relations(code, all_edges)

    if backend == "plotly":
        return _plot_hasse_plotly(code, covers)

    fig, ax_out = prepare_axes(ax, interactive=interactive)
    _plot_hasse_matplotlib(code, covers, ax_out)
    if ax is not None:
        return ax_out
    return fig


def _plot_hasse_matplotlib(code, covers, ax):
    G = nx.DiGraph()
    G.add_nodes_from(code)
    G.add_edges_from(covers)

    layers = {}
    for c in code:
        w = _hamming_weight(c)
        layers.setdefault(w, []).append(c)

    pos = {}
    max_weight = max(layers.keys()) if layers else 0
    for weight, nodes in layers.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2) * 1.5
            pos[node] = (x, weight)

    max_w = max_weight if max_weight > 0 else 1
    node_colors = []
    cmap = plt.cm.viridis
    for c in G.nodes():
        w = _hamming_weight(c)
        node_colors.append(cmap(w / max_w))

    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=800, font_size=8, font_weight="bold", arrows=False,
            edge_color="#666666", width=1.5)
    ax.set_title("Hasse Diagram", fontsize=12)


def _plot_hasse_plotly(code, covers):
    import plotly.graph_objects as go

    layers = {}
    for c in code:
        w = _hamming_weight(c)
        layers.setdefault(w, []).append(c)

    pos = {}
    for weight, nodes in layers.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = ((i - (n - 1) / 2) * 1.5, weight)

    edge_x, edge_y = [], []
    for c1, c2 in covers:
        x0, y0 = pos[c1]
        x1, y1 = pos[c2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1.5, color="#666"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in code], y=[pos[c][1] for c in code],
        mode="markers+text",
        marker=dict(size=20, color=[_hamming_weight(c) for c in code],
                    colorscale="Viridis", line=dict(width=1, color="#333")),
        text=code, textposition="top center",
        hovertext=[f"{c} (weight {_hamming_weight(c)})" for c in code],
    ))
    fig.update_layout(title="Hasse Diagram", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig
