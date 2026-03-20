import pytest
from neuralviz._core import (
    normalize_input,
    validate_code,
    neuron_colors,
    get_backend,
    prepare_axes,
)

class TestNormalizeInput:
    def test_list_of_strings(self):
        assert normalize_input(["001", "010", "110"]) == ["001", "010", "110"]
    def test_tuple_of_strings(self):
        assert normalize_input(("001", "010")) == ["001", "010"]
    def test_sage_neuralcode_duck_type(self):
        class FakeNeuralCode:
            Codes = ["001", "010", "110"]
        assert normalize_input(FakeNeuralCode()) == ["001", "010", "110"]

class TestValidateCode:
    def test_valid_code(self):
        validate_code(["001", "010", "110"])
    def test_empty_code_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_code([])
    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            validate_code(["001", "01"])
    def test_non_binary_raises(self):
        with pytest.raises(ValueError, match="binary"):
            validate_code(["002", "010"])

class TestNeuronColors:
    def test_returns_correct_count(self):
        colors = neuron_colors(3)
        assert len(colors) == 3
    def test_canonical_order_first_three(self):
        colors = neuron_colors(3)
        assert colors[0] == "#FF0000"
        assert colors[1] == "#0000FF"
        assert colors[2] == "#FFFF00"
    def test_wraps_with_brightness_shift_beyond_8(self):
        colors = neuron_colors(10)
        assert len(colors) == 10
        assert colors[8] != colors[0]
        assert colors[9] != colors[1]

class TestGetBackend:
    def test_matplotlib_default(self):
        assert get_backend(False) == "matplotlib"
    def test_plotly_when_available(self):
        try:
            import plotly
            assert get_backend(True) == "plotly"
        except ImportError:
            with pytest.raises(ImportError, match="plotly"):
                get_backend(True)

class TestPrepareAxes:
    def test_creates_new_figure_when_ax_none(self):
        import matplotlib.pyplot as plt
        fig, ax = prepare_axes(None)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    def test_uses_provided_ax(self):
        import matplotlib.pyplot as plt
        fig, ax_orig = plt.subplots()
        fig_out, ax_out = prepare_axes(ax_orig)
        assert ax_out is ax_orig
        assert fig_out is fig
        plt.close(fig)
    def test_raises_if_interactive_and_ax(self):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="interactive"):
            prepare_axes(ax, interactive=True)
        plt.close("all")
