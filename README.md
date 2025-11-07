
# spacetime-lib

Vibecoded with ChatGPT using GPT-5.

A lightweight, object-oriented wrapper around SymPy for common GR computations
in coordinates: metrics, Christoffel symbols, Ricci tensor, and Ricci scalar.

## Install / Use

```python
import sympy as sp
from spacetime import Spacetime

theta, phi = sp.symbols('theta phi', real=True)
dtheta, dphi = sp.symbols('dtheta dphi', real=True)

# Unit 2-sphere
ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
S2 = Spacetime((theta, phi), ds2)

# Access read-only attributes
Gamma = S2.Gamma           # Γ^i_{jk}
Ricci = S2.Ricci           # R_ij
R     = S2.Ricci_scalar    # scalar curvature

# Pretty-print non-zeros
S2.print_nonzero()
S2.print_metric()
```

## Coordinate changes

Transform to a new chart using old coordinates expressed as functions of the new:

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

## Notes

- The line element must use symbols named `d<coord>` to represent differentials.
- Metrics with off-diagonal terms are supported.
- All computations assume the Levi-Civita connection.
- Symbolic simplification is applied; use SymPy's `simplify`/`factor` etc. as needed.


### Riemann tensor
- `Spacetime.Riemann` returns `R[i,j,k,l] = R^i_{ j k l }`.
- `Spacetime.Ricci` contracts `Riemann`: `R_ij = R^k_{ i k j }`.
- `print_nonzero(..., show_christoffel=True, show_riemann=True, show_ricci=True, show_scalar=True)`
  controls printed items.

## Examples

Run `python examples/schwarzschild_examples.py` to explore metric determinants, horizon locations, and representative components of the Schwarzschild solution in static, advanced Eddington–Finkelstein, and isotropic coordinates generated via `Spacetime.change_coordinates`.
