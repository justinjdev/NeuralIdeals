import pytest
import matplotlib.pyplot as plt
from neuralviz.receptive_fields import plot_receptive_fields, _extract_rf_relations

class TestExtractRfRelations:
    def test_disjoint(self):
        relations = _extract_rf_relations(["00", "10", "01"])
        assert any(set(d) == {0, 1} for d in relations.get("disjoint", []))

    def test_containment(self):
        relations = _extract_rf_relations(["00", "10", "11"])
        containments = relations.get("containments", [])
        assert any(set(sigma) == {1} and 0 in tau for sigma, tau in containments)

    def test_from_rf_structure_dict(self):
        rf = {"containments": [([1], [0])], "disjoint": [], "covers": []}
        fig = plot_receptive_fields(["00", "10", "11"], rf_structure=rf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

class TestPlotReceptiveFields:
    def test_returns_figure(self):
        fig = plot_receptive_fields(["001", "010", "110"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self):
        fig, ax = plt.subplots()
        result = plot_receptive_fields(["10", "01"], ax=ax)
        assert result is ax
        plt.close(fig)

    def test_all_fire_together(self):
        fig = plot_receptive_fields(["000", "111"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
