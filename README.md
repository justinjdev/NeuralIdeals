# NeuralIdeals

Computational tools for analyzing and visualizing [neural codes](https://arxiv.org/abs/1609.09602) — algebraic objects that encode how neurons represent spatial information through receptive fields.

## Packages

### neuralviz (pure Python)

Visualization package for neural codes. No SageMath or Java required.

```bash
pip install git+https://github.com/justinjdev/NeuralIdeals.git
```

```python
from neuralviz import (
    plot_hasse,               # Hasse diagram of codeword poset
    plot_simplicial_complex,  # nerve complex
    plot_receptive_fields,    # Euler/Venn RF overlap diagrams
    plot_convex_1d,           # interval realization in R¹
    plot_convex_2d,           # polygon realization in R² (experimental)
    plot_arcs,                # circular arc display
    plot_lines,               # grid line display
)

code = ['001', '010', '110']
plot_hasse(code)
plot_simplicial_complex(code)
plot_receptive_fields(code)
plot_convex_1d(code)
```

All functions return matplotlib figures by default. Pass `interactive=True` for plotly.

**Optional dependencies:**

```bash
# Interactive plotly output
pip install "neuralviz[interactive] @ git+https://github.com/justinjdev/NeuralIdeals.git"

# R² convex realization (shapely + scipy)
pip install "neuralviz[all] @ git+https://github.com/justinjdev/NeuralIdeals.git"
```

### neuralcode.py (SageMath)

Algebraic computation engine for neural ideals. Requires [SageMath](https://www.sagemath.org/).

```python
load('load_all.py')

nc = NeuralCode(['001', '010', '110'])
nc.canonical()                  # canonical form of the neural ideal
nc.factored_canonical()         # factored generators
nc.canonical_RF_structure()     # receptive field structure
nc.groebner_basis()             # Groebner basis
```

## Paper

Based on:

> Ethan Petersen, Nora Youngs, Ryan Kruse, Dane Miyata, Rebecca Garcia, Luis David Garcia Puente.
> **Neural Ideals in SageMath.** [arXiv:1609.09602](https://arxiv.org/abs/1609.09602), 2016.

## License

MIT
