"""Render a short LaTeX report for a 2-sphere metric."""

from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacetime import Spacetime


def main() -> None:
    theta, phi = sp.symbols("theta phi", real=True)
    dtheta, dphi = sp.symbols("dtheta dphi", real=True)
    ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
    s2 = Spacetime((theta, phi), ds2)

    output_stem = Path("reports") / "two_sphere"
    pdf_path = s2.render_latex_pdf(
        output_stem,
        show_riemann=False,  # keep the PDF readable
        show_all_pairs=True,
    )
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
