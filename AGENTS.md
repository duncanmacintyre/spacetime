# Repository Guidelines

## Project Structure & Module Organization
The core library lives in `spacetime/spacetime.py`, exposing the `Spacetime` dataclass through `spacetime/__init__.py`. Helper functions prefixed with underscores stay in the same module to keep the public API narrow. Tests reside in `tests/test_spacetime.py`; reuse that file as the template when adding new regression cases. Keep experimental notebooks or scratchpads outside the package—create an `examples/` directory if enduring assets are needed.

## Build, Test, and Development Commands
Create a virtual environment and install SymPy with `python -m pip install sympy`. Run the full suite via `python -m unittest discover tests -v`. During iteration you can target a single case, e.g. `python -m unittest tests.test_spacetime.TestSpacetimeBasics.test_two_sphere`. When exploring in a REPL, add the repo root to `PYTHONPATH` (mirroring the tests) until packaging metadata is introduced.

## Coding Style & Naming Conventions
Follow standard PEP 8 defaults: 4-space indentation, descriptive snake_case for functions (`_ricci_scalar`), and CapWords for classes (`Spacetime`). Keep internal helpers module-private unless they are part of the supported API. Favor explicit SymPy constructors (`sp.Symbol`, `sp.Matrix`) and call `sp.simplify(...)` before returning derived quantities. Prefer docstrings for public entry points; reserve inline comments for non-obvious symbolic manipulations.

## Testing Guidelines
Tests use the standard library `unittest` runner. Name files `test_*.py` and group related assertions inside `unittest.TestCase` subclasses. Mirror the existing pattern of symbolic setup, transformation, and comparison against simplified expectations. Ensure new features include at least one curvature or tensor regression check, and rerun `python -m unittest` before opening a PR. If a calculation is expensive, guard it with targeted assertions so the suite stays under a minute.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit summaries (`Add initial generated code`). Keep bodies concise but mention any symbolic identities or performance trade-offs. Pull requests should describe the change set, list new commands or dependencies, and note how curvature invariants behave after the change. Link issues when available and paste the `python -m unittest` output when proposing merges.

## SymPy & Coordinate Tips
Coordinate symbols must pair with differentials named `d<coord>` for `_metric_from_line_element` to detect metric components. When introducing new charts, provide `old_as_functions_of_new` tuples that match the established ordering, and call `sp.simplify` or `sp.together` to stabilize regression expectations.
