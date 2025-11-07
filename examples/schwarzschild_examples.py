"""
Example computations for the Schwarzschild spacetime in multiple charts.

The script highlights how to instantiate `Spacetime`, inspect derived metric
data (determinants, horizon locations, notable components), and use
`change_coordinates` to explore alternative coordinate systems such as
advanced Eddington–Finkelstein and isotropic coordinates. Run it directly to
print summaries of the resulting charts.
"""

from __future__ import annotations

import sympy as sp

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from spacetime import Spacetime


# Global symbols shared across the examples
M = sp.symbols("M", positive=True)
t, r, theta, phi = sp.symbols("t r theta phi", real=True)
dt, dr, dtheta, dphi = sp.symbols("dt dr dtheta dphi", real=True)


def schwarzschild_static() -> Spacetime:
    """Return the Schwarzschild solution in standard static coordinates."""
    f = 1 - (2 * M) / r
    ds2 = (
        f * dt ** 2
        - f ** -1 * dr ** 2
        - r ** 2 * (dtheta ** 2 + sp.sin(theta) ** 2 * dphi ** 2)
    )
    return Spacetime((t, r, theta, phi), ds2)


def to_advanced_eddington_finkelstein(metric: Spacetime) -> Spacetime:
    """
    Transform the Schwarzschild metric to advanced Eddington–Finkelstein
    coordinates (v, r, theta, phi) via `change_coordinates`.
    """
    v = sp.symbols("v", real=True)

    # Retain the areal radius and angles, rewrite t in terms of (v, r).
    tortoise = r + 2 * M * sp.log(sp.Abs(r - 2 * M))
    t_expr = v - tortoise
    return metric.change_coordinates(
        new_coords=(v, r, theta, phi),
        old_as_functions_of_new=(t_expr, r, theta, phi),
        simplify=False,
    )


def to_isotropic(metric: Spacetime) -> Spacetime:
    """
    Transform the Schwarzschild solution to isotropic coordinates
    (T, rho, theta, phi) where the spatial slices are conformally flat.
    """
    T, rho = sp.symbols("T rho", real=True)

    # Old coordinates expressed through the new symbols.
    r_expr = rho * (1 + M / (2 * rho)) ** 2
    return metric.change_coordinates(
        new_coords=(T, rho, theta, phi),
        old_as_functions_of_new=(T, r_expr, theta, phi),
        simplify=False,
    )


def summarize(
    metric: Spacetime,
    label: str,
    *,
    horizon_variable: sp.Symbol,
    metric_components: list[tuple[int, int]] | None = None,
    show_metric_matrix: bool = False,
) -> None:
    """Pretty-print diagnostic data for a given chart."""
    print(f"=== {label} ===")
    print("Coordinates:", metric.coordinates)
    g_det = sp.simplify(metric.metric.det())
    print("det(g):", g_det)
    g_00 = sp.simplify(metric.metric[0, 0])
    print("Metric component g_00:", g_00)
    horizon = sp.solve(sp.Eq(g_00, 0), horizon_variable, dict=True)
    print(f"Event horizon solutions ({horizon_variable}):", horizon)
    if metric_components:
        for i, j in metric_components:
            print(f"Metric component g_{i}{j}:", sp.simplify(metric.metric[i, j]))
    if show_metric_matrix:
        print("Metric tensor g_ij:")
        sp.pprint(metric.metric)
    print()


def main() -> None:
    base = schwarzschild_static()
    summarize(
        base,
        "Schwarzschild (t, r, theta, phi)",
        horizon_variable=r,
        metric_components=[(1, 1)],
    )

    ef = to_advanced_eddington_finkelstein(base)
    summarize(
        ef,
        "Advanced Eddington–Finkelstein (v, r, theta, phi)",
        horizon_variable=r,
        metric_components=[(0, 1), (1, 1)],
    )

    iso = to_isotropic(base)
    summarize(
        iso,
        "Isotropic (T, rho, theta, phi)",
        horizon_variable=iso.coordinates[1],
        metric_components=[(1, 1)],
    )


if __name__ == "__main__":
    main()
