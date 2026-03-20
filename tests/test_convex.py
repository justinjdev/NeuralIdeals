import pytest
import matplotlib.pyplot as plt
from neuralviz.convex_realization import (
    plot_convex_1d, _compute_intervals, _verify_realization, _find_valid_ordering,
)

class TestFindValidOrdering:
    def test_single_neuron(self):
        ordering = _find_valid_ordering(["0", "1"])
        assert ordering is not None
        assert set(ordering) == {"0", "1"}

    def test_contiguous_result(self):
        code = ["00", "10", "01", "11"]
        ordering = _find_valid_ordering(code)
        assert ordering is not None
        for neuron in range(2):
            indices = [i for i, c in enumerate(ordering) if c[neuron] == "1"]
            assert indices[-1] - indices[0] == len(indices) - 1

class TestComputeIntervals:
    def test_single_neuron(self):
        intervals = _compute_intervals(["0", "1"])
        assert len(intervals) == 1
        a, b = intervals[0]
        assert a < b

    def test_two_overlapping(self):
        intervals = _compute_intervals(["00", "10", "01", "11"])
        assert len(intervals) == 2

    def test_two_disjoint(self):
        intervals = _compute_intervals(["00", "10", "01"])
        assert len(intervals) == 2
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        assert sorted_intervals[0][1] <= sorted_intervals[1][0]

class TestVerifyRealization:
    def test_correct_realization(self):
        code = ["00", "10", "01", "11"]
        intervals = _compute_intervals(code)
        assert _verify_realization(code, intervals)

    def test_single_neuron(self):
        code = ["0", "1"]
        intervals = _compute_intervals(code)
        assert _verify_realization(code, intervals)

    def test_round_trip_paper_example(self):
        code = ["000", "100", "010", "110", "011", "111"]
        intervals = _compute_intervals(code)
        assert _verify_realization(code, intervals)

    def test_checks_midpoints(self):
        code = ["00", "10", "01"]
        intervals = _compute_intervals(code)
        assert _verify_realization(code, intervals)

class TestPlotConvex1d:
    def test_returns_figure(self):
        fig = plot_convex_1d(["00", "10", "01", "11"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self):
        fig, ax = plt.subplots()
        result = plot_convex_1d(["0", "1"], ax=ax)
        assert result is ax
        plt.close(fig)

    def test_paper_example(self):
        fig = plot_convex_1d(["001", "010", "110"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_three_neurons_overlapping(self):
        fig = plot_convex_1d(["000", "100", "010", "110", "011", "111"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
