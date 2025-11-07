
# spacetime-lib

Vibecoded with ChatGPT using GPT-5.

`spacetime` is a lightweight, object-oriented wrapper around SymPy for common
GR computations in coordinates—metrics, Christoffel symbols, Ricci tensor, and
Ricci scalar. It also has a few higher-level utilities for coordinate changes
and generating PDFs that show components of the Christoffel symbols, Ricci
tensor, and Ricci scalar.

## Quick start

```python
import sympy as sp
from spacetime import Spacetime

theta, phi = sp.symbols('theta phi', real=True)
dtheta, dphi = sp.symbols('dtheta dphi', real=True)

# Unit 2-sphere
ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
S2 = Spacetime((theta, phi), ds2)

# Cached tensor accessors
Gamma = S2.Gamma          # Γ^i_{jk}
Riemann = S2.Riemann      # R^i_{ jkl }
Ricci = S2.Ricci          # R_ij
scalar = S2.Ricci_scalar  # curvature scalar R

S2.print_metric()       # ds^2 in terms of the cached metric
S2.print_nonzero()      # summarized tensors, defaulting to plain text
```

## Printing descriptions of the spacetime

`print_metric()` prints the line element in
terms of the canonical `d<coord>` symbols. Inside IPython/Jupyter it
automatically routes the result through MathJax; otherwise it falls back to
plain text.

`print_nonzero` prints out non-zero elements of Christoffel symbols, Riemann
tensors, Ricci tensors, and Ricci scalars. It now accepts ``latex=None`` by
default:

- ``latex=None`` – auto mode. In IPython/Jupyter the tensors render via
  MathJax; in plain terminals they fall back to text.
- ``latex=True`` – always print LaTeX strings (no auto-rendering) so you can
  copy/paste into a document.
- ``latex=False`` – always print plain SymPy strings.

All parameters are optional; the snippet shows their default values plus a brief
description:

```python
S2.print_nonzero(
    latex=None,                # Auto; True prints LaTeX text, False plain text
    show_all_pairs=False,      # Only j<=k Christoffel symbols; True shows all
    show_christoffel=True,     # Include Γ^i_{jk}
    show_riemann=True,         # Include R^i_{ jkl }
    show_ricci=True,           # Include R_ij
    show_scalar=True,          # Include Ricci scalar R
)
```

### Notebook-friendly LaTeX

`Spacetime.latex_components` has the same arguments but returns a string you can
feed directly to IPython:

```python
from IPython.display import Math

Math(
    S2.latex_components(
        show_riemann=False,   # hide bulky tensors if desired
        show_all_pairs=True,  # include all Γ^i_{jk}
        # other arguments as above
    )
)
```

## Exporting PDFs

To generate a PDF describing the spacetime, call `Spacetime.render_latex_pdf`.
It mirrors the usage of `print_nonzero` but adds a few output controls. 

```python
pdf_path = S2.render_latex_pdf(
    filename="spacetime_report",  # Output stem; .tex/.pdf live next to it
    show_metric=True,              # Include ds^2 block
    show_christoffel=True,         # Include Γ^i_{jk}
    show_riemann=True,             # Include R^i_{ jkl }
    show_ricci=True,               # Include R_ij
    show_scalar=True,              # Include Ricci scalar R
    show_all_pairs=False,          # Upper-triangle Γ unless True
    pdflatex="pdflatex",          # LaTeX engine invoked to build the PDF
    cleanup_auxiliary=True,        # Remove .aux/.log/.out afterward
)
print(f"PDF ready at {pdf_path}")
```

- Output: `spacetime_report.tex` and `spacetime_report.pdf` (or whatever stem
  you provide). Directories are created automatically.
- Requirements: a LaTeX distribution available on `PATH` matching the
  `pdflatex` argument.
- Clean-up: auxiliary files vanish by default; set `cleanup_auxiliary=False`
  to inspect them.

## Coordinate changes

Transform to a new Spacetime using old coordinates expressed as functions of the new:

```python
x, y = sp.symbols('x y', real=True)
dx, dy = sp.symbols('dx dy', real=True)
ds2_cart = dx**2 + dy**2
E2 = Spacetime((x, y), ds2_cart)

r, phi = sp.symbols('r phi', real=True)
E2_polar = E2.change_coordinates(
    new_coords=(r, phi),
    old_as_functions_of_new=(r*sp.cos(phi), r*sp.sin(phi))
)
E2_polar.print_metric()  # ds^2 = dr^2 + r^2 dphi^2
```

## Additional notes

- Line elements must use symbols named `d<coord>` so the metric reconstruction
  logic can match them to `coords`. (Example: `dx`, `dy`, `dt` for `x`, `y`, `t`.)
- Off-diagonal metrics are fully supported; `_metric_from_line_element` handles
  mixed differential terms automatically.
- Everything assumes a Levi-Civita connection. Switch to `sp.diff`/`sp.Matrix`
  yourself if you need torsionful connections.
- SymPy simplification (`sp.simplify`) is applied internally before storing
  tensors. For heavier algebraic manipulation, compose with SymPy tools
  (`simplify`, `factor`, `together`, etc.).

### The Riemanns and Riccis as SymPy objects

- `Spacetime.Riemann` returns `R^i_{ j k l }` as a rank-4 SymPy array.
- `Spacetime.Ricci` contracts as `R_ij = R^k_{ i j k }`; because of the selected
  index placement, a positively curved 2-sphere yields `Ricci = -metric`.
- `Spacetime.Ricci_scalar` gives the curvature scalar `R`.

## Examples

- `python examples/schwarzschild_examples.py` explores metric determinants,
  horizons, and representative components of the Schwarzschild solution in
  multiple charts using `Spacetime.change_coordinates`.
- `python examples/render_pdf_report.py` shows how to generate a 2-sphere PDF
  using the `render_latex_pdf` helper.
- `examples/Schwarzschild notebook.ipynb` demonstrates usage in a Jupyter
  notebook, including MathJax-rendered tensors.
