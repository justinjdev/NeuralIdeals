import pytest
import matplotlib.pyplot as plt
from neuralviz.legacy import plot_arcs, plot_lines, plot_circles

class TestPlotLines:
    def test_returns_figure(self):
        fig = plot_lines(["100", "110", "011"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self):
        fig, ax = plt.subplots()
        result = plot_lines(["10", "01"], ax=ax)
        assert result is ax
        plt.close(fig)

    def test_single_codeword(self):
        fig = plot_lines(["101"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

class TestPlotArcs:
    def test_returns_figure(self):
        fig = plot_arcs(["100", "110", "011"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self):
        fig, ax = plt.subplots()
        result = plot_arcs(["10", "01"], ax=ax)
        assert result is ax
        plt.close(fig)

    def test_single_codeword(self):
        fig = plot_arcs(["111"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

class TestPlotCircles:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            plot_circles(["110", "011"])
