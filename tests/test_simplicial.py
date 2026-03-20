import pytest
import matplotlib.pyplot as plt
from neuralviz.simplicial_complex import (
    plot_simplicial_complex, _extract_simplices, _maximal_simplices,
)

class TestExtractSimplices:
    def test_basic(self):
        code = ["110", "010", "001"]
        simplices = _extract_simplices(code)
        assert frozenset({0, 1}) in simplices
        assert frozenset({1}) in simplices
        assert frozenset({2}) in simplices

    def test_all_zeros_excluded(self):
        code = ["000", "100"]
        simplices = _extract_simplices(code)
        assert frozenset() not in simplices
        assert frozenset({0}) in simplices

    def test_full_simplex(self):
        simplices = _extract_simplices(["111"])
        assert frozenset({0, 1, 2}) in simplices

class TestMaximalSimplices:
    def test_removes_faces(self):
        simplices = {frozenset({0, 1}), frozenset({0}), frozenset({1}), frozenset({2})}
        maximal = _maximal_simplices(simplices)
        assert frozenset({0, 1}) in maximal
        assert frozenset({2}) in maximal
        assert frozenset({0}) not in maximal

class TestPlotSimplicialComplex:
    def test_returns_figure(self):
        fig = plot_simplicial_complex(["110", "010", "001"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self):
        fig, ax = plt.subplots()
        result = plot_simplicial_complex(["110", "001"], ax=ax)
        assert result is ax
        plt.close(fig)

    def test_triangle_face(self):
        fig = plot_simplicial_complex(["111", "110", "101", "011", "100", "010", "001"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
