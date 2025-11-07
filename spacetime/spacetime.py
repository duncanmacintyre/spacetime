
"""
spacetime.spacetime

Object-oriented GR helpers on top of SymPy.
"""
from __future__ import annotations
import sympy as sp
from dataclasses import dataclass
from typing import Sequence, Tuple

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

    def print_metric(self):
        dcoords = [_find_symbol_by_name(self.ds2, f"d{str(c)}") for c in self._coords]
        ds2 = 0
        n = len(self._coords)
        for i in range(n):
            for j in range(n):
                ds2 += self._g[i,j]*dcoords[i]*dcoords[j]
        print(f"ds^2 = {sp.sstr(sp.simplify(ds2))}")

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
