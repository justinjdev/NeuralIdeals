import pytest
import matplotlib.pyplot as plt
from neuralviz.hasse import plot_hasse, _build_poset, _covering_relations

class TestBuildPoset:
    def test_total_order(self):
        code = ["000", "100", "110"]
        edges = _build_poset(code)
        assert ("000", "100") in edges
        assert ("000", "110") in edges
        assert ("100", "110") in edges

    def test_no_comparable_pair(self):
        code = ["100", "010"]
        edges = _build_poset(code)
        assert len(edges) == 0

    def test_single_codeword(self):
        edges = _build_poset(["101"])
        assert len(edges) == 0

class TestCoveringRelations:
    def test_removes_transitive_edges(self):
        code = ["000", "100", "110"]
        all_edges = _build_poset(code)
        covers = _covering_relations(code, all_edges)
        assert ("000", "100") in covers
        assert ("100", "110") in covers
        assert ("000", "110") not in covers

    def test_diamond(self):
        code = ["000", "100", "010", "110"]
        all_edges = _build_poset(code)
        covers = _covering_relations(code, all_edges)
        assert ("000", "100") in covers
        assert ("000", "010") in covers
        assert ("100", "110") in covers
        assert ("010", "110") in covers
        assert ("000", "110") not in covers

class TestPlotHasse:
    def test_returns_figure(self):
        fig = plot_hasse(["000", "100", "010", "110"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self):
        fig, ax = plt.subplots()
        result = plot_hasse(["000", "100"], ax=ax)
        assert result is ax
        plt.close(fig)

    def test_single_codeword(self):
        fig = plot_hasse(["101"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_paper_example(self):
        fig = plot_hasse(["001", "010", "110"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
