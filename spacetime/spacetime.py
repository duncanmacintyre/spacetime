
"""
spacetime.spacetime

Object-oriented GR helpers on top of SymPy.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
from textwrap import dedent
from typing import Sequence, Tuple, Union

import sympy as sp

__all__ = ["Spacetime"]

def _find_symbol_by_name(expr: sp.Expr, name: str) -> sp.Symbol:
    for s in expr.free_symbols:
        if str(s) == name:
            return s
    return sp.Symbol(name)

def _metric_from_line_element(ds2: sp.Expr, coords: Sequence[sp.Symbol]) -> sp.Matrix:
    n = len(coords)
    dcoords = [_find_symbol_by_name(ds2, f"d{str(c)}") for c in coords]
    expr = sp.expand(ds2)
    g = sp.MutableDenseMatrix([[sp.S.Zero]*n for _ in range(n)])
    for i in range(n):
        g[i, i] = sp.expand(expr).coeff(dcoords[i]**2)
    for i in range(n):
        for j in range(i+1, n):
            coeff = sp.expand(expr).coeff(dcoords[i]*dcoords[j])
            g[i, j] = coeff/2
            g[j, i] = coeff/2
    return sp.Matrix(g)

def _christoffel(g: sp.Matrix, coords: Sequence[sp.Symbol], simplify: bool=True) -> sp.MutableDenseNDimArray:
    n = len(coords)
    g_inv = sp.simplify(g.inv()) if simplify else g.inv()
    Gamma = sp.MutableDenseNDimArray([0]*(n**3), (n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(j, n):
                term = 0
                for l in range(n):
                    term += g_inv[i,l]*(sp.diff(g[l,k], coords[j]) +
                                        sp.diff(g[l,j], coords[k]) -
                                        sp.diff(g[j,k], coords[l]))
                term = sp.Rational(1,2)*term
                if simplify:
                    term = sp.simplify(term)
                Gamma[i,j,k] = term
                Gamma[i,k,j] = term
    return Gamma

def _riemann(Gamma: sp.MutableDenseNDimArray, coords: Sequence[sp.Symbol], simplify: bool=True) -> sp.MutableDenseNDimArray:
    """
    R^i_{ j k l } = ∂_k Γ^i_{j l} - ∂_l Γ^i_{j k} + Γ^i_{m k} Γ^m_{j l} - Γ^i_{m l} Γ^m_{j k}
    """
    n = len(coords)
    R = sp.MutableDenseNDimArray([0]*(n**4), (n,n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    term = sp.diff(Gamma[i,j,l], coords[k]) - sp.diff(Gamma[i,j,k], coords[l])
                    s1 = 0
                    s2 = 0
                    for m in range(n):
                        s1 += Gamma[i,m,k]*Gamma[m,j,l]
                        s2 += Gamma[i,m,l]*Gamma[m,j,k]
                    term += s1 - s2
                    R[i,j,k,l] = sp.simplify(term) if simplify else term
    return R

def _ricci_from_riemann(Riemann: sp.MutableDenseNDimArray, simplify: bool=True) -> sp.Matrix:
    """R_ij = R^k_{ i j k }"""
    n = Riemann.shape[0]
    R = sp.MutableDenseMatrix([[sp.S.Zero]*n for _ in range(n)])
    for i in range(n):
        for j in range(n):
            term = 0
            for k in range(n):
                term += Riemann[k, i, j, k]
            R[i,j] = sp.simplify(term) if simplify else term
    return sp.Matrix(R)

def _ricci_scalar(R: sp.Matrix, g_inv: sp.Matrix, simplify: bool=True) -> sp.Expr:
    n = R.shape[0]
    expr = 0
    for i in range(n):
        for j in range(n):
            expr += g_inv[i,j]*R[i,j]
    return sp.simplify(expr) if simplify else expr

@dataclass(frozen=True)
class Spacetime:
    coords: Tuple[sp.Symbol, ...]
    ds2: sp.Expr

    def __post_init__(self):
        coords = tuple(self.coords)
        object.__setattr__(self, "_coords", coords)
        g = _metric_from_line_element(self.ds2, coords)
        object.__setattr__(self, "_g", sp.simplify(g))
        object.__setattr__(self, "_g_inv", None)
        object.__setattr__(self, "_Gamma", None)
        object.__setattr__(self, "_Riemann", None)
        object.__setattr__(self, "_Ricci", None)
        object.__setattr__(self, "_Ricci_scalar", None)

    @property
    def coordinates(self) -> Tuple[sp.Symbol, ...]:
        return self._coords

    @property
    def metric(self) -> sp.Matrix:
        return self._g

    @property
    def inv_metric(self) -> sp.Matrix:
        if self._g_inv is None:
            object.__setattr__(self, "_g_inv", sp.simplify(self._g.inv()))
        return self._g_inv

    @property
    def Gamma(self) -> sp.MutableDenseNDimArray:
        if self._Gamma is None:
            object.__setattr__(self, "_Gamma", _christoffel(self._g, self._coords, simplify=True))
        return self._Gamma

    @property
    def Riemann(self) -> sp.MutableDenseNDimArray:
        if self._Riemann is None:
            object.__setattr__(self, "_Riemann", _riemann(self.Gamma, self._coords, simplify=True))
        return self._Riemann

    @property
    def Ricci(self) -> sp.Matrix:
        if self._Ricci is None:
            object.__setattr__(self, "_Ricci", _ricci_from_riemann(self.Riemann, simplify=True))
        return self._Ricci

    @property
    def Ricci_scalar(self) -> sp.Expr:
        if self._Ricci_scalar is None:
            s = _ricci_scalar(self.Ricci, self.inv_metric, simplify=True)
            object.__setattr__(self, "_Ricci_scalar", s)
        return self._Ricci_scalar

    def change_coordinates(self, new_coords, old_as_functions_of_new, simplify: bool=True) -> "Spacetime":
        if len(old_as_functions_of_new) != len(self._coords):
            raise ValueError("Provide one expression for each old coordinate.")
        n = len(self._coords)
        J = sp.MutableDenseMatrix([[sp.diff(old_as_functions_of_new[i], new_coords[a])
                                    for a in range(n)] for i in range(n)])
        subs_map = {self._coords[i]: old_as_functions_of_new[i] for i in range(n)}
        g_sub = sp.Matrix([[sp.simplify(self._g[i,j].subs(subs_map)) for j in range(n)] for i in range(n)])
        gprime = (J.T) * g_sub * J
        if simplify:
            gprime = sp.simplify(sp.Matrix(gprime))
        dnew = [sp.Symbol(f"d{str(c)}") for c in new_coords]
        ds2_new = 0
        for a in range(n):
            for b in range(n):
                ds2_new += gprime[a,b]*dnew[a]*dnew[b]
        return Spacetime(tuple(new_coords), sp.expand(ds2_new))

    def _differential_symbols(self) -> Tuple[sp.Symbol, ...]:
        return tuple(_find_symbol_by_name(self.ds2, f"d{str(c)}") for c in self._coords)

    def _line_element_expr(self) -> sp.Expr:
        dcoords = self._differential_symbols()
        ds2 = sp.S.Zero
        n = len(self._coords)
        for i in range(n):
            for j in range(n):
                ds2 += self._g[i,j]*dcoords[i]*dcoords[j]
        return sp.simplify(ds2)

    def _latex_symbol_map(self) -> dict[sp.Symbol, str]:
        mapping = {}
        for coord, dcoord in zip(self._coords, self._differential_symbols()):
            mapping[dcoord] = rf"d{sp.latex(coord)}"
        return mapping

    def print_metric(self):
        ds2 = self._line_element_expr()
        print(f"ds^2 = {sp.sstr(ds2)}")

    def print_nonzero(self, latex: bool=False, show_all_pairs: bool=False,
                      show_christoffel: bool=True, show_riemann: bool=True,
                      show_ricci: bool=True, show_scalar: bool=True):
        coords = self._coords
        n = len(coords)
        zero = sp.S.Zero

        def fmt(expr):
            return sp.latex(sp.simplify(expr)) if latex else sp.sstr(sp.simplify(expr))

        # Christoffels
        if show_christoffel:
            print("# Non-zero Christoffel symbols Γ^i_{jk}")
        printed = 0
        for i in range(n):
            for j in range(n):
                k_iter = range(n) if show_all_pairs else range(j, n)
                for k in k_iter:
                    val = sp.simplify(self.Gamma[i, j, k])
                    if val != zero:
                        if latex:
                            print(rf"\Gamma^{i}_{{{j}{k}}} = {fmt(val)}")
                        else:
                            print(f"Gamma^{i}_{j}{k} = {fmt(val)}")
                        printed += 1
        if show_christoffel and printed == 0:
            print("(all Γ vanish)")

        # Riemann
        if show_riemann:
            print("# Non-zero Riemann tensor components R^i_{ j k l }")
            printed_Riem = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            val = sp.simplify(self.Riemann[i,j,k,l])
                            if val != zero:
                                if latex:
                                    print(rf"R^{i}_{{{j}{k}{l}}} = {fmt(val)}")
                                else:
                                    print(f"R^{i}_{j}{k}{l} = {fmt(val)}")
                                printed_Riem += 1
            if printed_Riem == 0:
                print("(all Riemann components vanish)")

        # Ricci
        printed_R = 0
        if show_ricci:
            print("# Non-zero Ricci tensor components R_ij")
        for i in range(n):
            for j in range(i, n):
                val = sp.simplify(self.Ricci[i,j])
                if val != zero and show_ricci:
                    print(f"R_{i}{j} = {fmt(val)}")
                    printed_R += 1
        if show_ricci and printed_R == 0:
            print("(all R_ij vanish)")

        # Scalar
        if show_scalar:
            print("# Ricci scalar R")
            print(fmt(self.Ricci_scalar))

    def render_latex_pdf(self, filename: Union[str, Path] = "spacetime_report",
                         show_metric: bool = True,
                         show_christoffel: bool = True,
                         show_riemann: bool = True,
                         show_ricci: bool = True,
                         show_scalar: bool = True,
                         show_all_pairs: bool = False,
                         pdflatex: str = "pdflatex",
                         cleanup_auxiliary: bool = True) -> Path:
        """Render a LaTeX report for this spacetime as a PDF.

        Parameters
        ----------
        filename:
            Target filename or stem for the resulting PDF. The matching .tex file
            is written in the same directory for inspection.
        show_metric/show_*:
            Mirror ``print_nonzero`` toggles to control which tensors appear in
            the report.
        show_all_pairs:
            Include off-diagonal Christoffel components instead of only the
            symmetric upper triangle.
        pdflatex:
            Executable used to turn the generated .tex file into a PDF.
        cleanup_auxiliary:
            Remove ``.aux``/``.log``/``.out`` files produced by ``pdflatex``
            after rendering completes.

        Returns
        -------
        Path
            Absolute path to the generated PDF.

        Raises
        ------
        RuntimeError
            If the LaTeX toolchain cannot be executed successfully.
        """

        output_path = Path(filename)
        if output_path.suffix:
            if output_path.suffix.lower() == ".pdf":
                pdf_path = output_path
            else:
                pdf_path = output_path.with_suffix(".pdf")
        else:
            pdf_path = output_path.parent / f"{output_path.name}.pdf"
        tex_path = pdf_path.with_suffix(".tex")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        latex_symbol_map = self._latex_symbol_map()

        def fmt(expr):
            return sp.latex(sp.simplify(expr), symbol_names=latex_symbol_map)
        lines = [r"\section*{Spacetime Report}"]
        coords_str = ", ".join(sp.latex(c) for c in self._coords)
        lines.append(rf"Coordinates: $({coords_str})$")

        def add_block(title: str, content: list[str]):
            if not content:
                return
            lines.append(rf"\subsection*{{{title}}}")
            lines.extend(content)

        if show_metric:
            ds2 = sp.latex(self._line_element_expr(), symbol_names=latex_symbol_map)
            add_block("Line element", [rf"\[ ds^2 = {ds2} \]"])

        n = len(self._coords)
        zero = sp.S.Zero

        if show_christoffel:
            christoffel_lines = []
            for i in range(n):
                for j in range(n):
                    k_iter = range(n) if show_all_pairs else range(j, n)
                    for k in k_iter:
                        val = sp.simplify(self.Gamma[i, j, k])
                        if val != zero:
                            christoffel_lines.append(
                                rf"\[ \Gamma^{{{i}}}_{{{j}{k}}} = {fmt(val)} \]"
                            )
            if not christoffel_lines:
                christoffel_lines.append(r"\[ \text{All Christoffel symbols vanish.} \]")
            add_block("Christoffel symbols", christoffel_lines)

        if show_riemann:
            riemann_lines = []
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            val = sp.simplify(self.Riemann[i,j,k,l])
                            if val != zero:
                                riemann_lines.append(
                                    rf"\[ R^{{{i}}}_{{{j}{k}{l}}} = {fmt(val)} \]"
                                )
            if not riemann_lines:
                riemann_lines.append(r"\[ \text{All Riemann tensor components vanish.} \]")
            add_block("Riemann tensor", riemann_lines)

        if show_ricci:
            ricci_lines = []
            for i in range(n):
                for j in range(i, n):
                    val = sp.simplify(self.Ricci[i,j])
                    if val != zero:
                        ricci_lines.append(rf"\[ R_{{{i}{j}}} = {fmt(val)} \]")
            if not ricci_lines:
                ricci_lines.append(r"\[ \text{All Ricci tensor components vanish.} \]")
            add_block("Ricci tensor", ricci_lines)

        if show_scalar:
            scalar_expr = fmt(self.Ricci_scalar)
            add_block("Ricci scalar", [rf"\[ R = {scalar_expr} \]"])

        document = dedent(
            f"""
            \\documentclass[11pt]{{article}}
            \\usepackage{{amsmath}}
            \\usepackage{{amssymb}}
            \\begin{{document}}
            {chr(10).join(lines)}
            \\end{{document}}
            """
        ).strip()
        tex_path.write_text(document)

        try:
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", tex_path.name],
                cwd=pdf_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Unable to find LaTeX engine '{pdflatex}'. Install a TeX distribution and retry."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
            raise RuntimeError(
                "pdflatex failed to render the spacetime report.\n" + stderr
            ) from exc

        if cleanup_auxiliary:
            for ext in (".aux", ".log", ".out"):
                aux_path = pdf_path.with_suffix(ext)
                if aux_path.exists():
                    aux_path.unlink()

        return pdf_path.resolve()
