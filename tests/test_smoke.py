"""Smoke tests: every public function runs without error on reference codes."""

import pytest
import matplotlib.pyplot as plt

from neuralviz import (
    plot_receptive_fields,
    plot_simplicial_complex,
    plot_convex_1d,
    plot_hasse,
    plot_arcs,
    plot_lines,
)

PAPER_CODES = [
    ["001", "010", "110"],
    ["000", "100", "010", "001", "110", "101", "011", "111"],
    ["000", "100", "010", "110"],
    ["110", "100", "000", "010"],
]

EDGE_CASES = [
    ["000"],
    ["111"],
    ["0", "1"],
]

ALL_CODES = PAPER_CODES + EDGE_CASES

PLOT_FUNCTIONS = [
    plot_receptive_fields,
    plot_simplicial_complex,
    plot_hasse,
    plot_arcs,
    plot_lines,
]


@pytest.mark.parametrize("plot_fn", PLOT_FUNCTIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("code", ALL_CODES, ids=lambda c: "-".join(c)[:20])
def test_smoke_static(plot_fn, code):
    fig = plot_fn(code)
    assert fig is not None
    plt.close("all")


@pytest.mark.parametrize("code", PAPER_CODES, ids=lambda c: "-".join(c)[:20])
def test_smoke_convex_1d(code):
    """plot_convex_1d may raise ValueError for non-realizable codes."""
    try:
        fig = plot_convex_1d(code)
        plt.close("all")
    except ValueError:
        pass


@pytest.mark.parametrize("code", PAPER_CODES[:2], ids=lambda c: "-".join(c)[:20])
def test_smoke_subplot_composition(code):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_simplicial_complex(code, ax=axes[0])
    plot_hasse(code, ax=axes[1])
    plot_lines(code, ax=axes[2])
    plt.close(fig)
