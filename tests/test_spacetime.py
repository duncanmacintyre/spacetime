
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from spacetime import Spacetime

class TestSpacetimeBasics(unittest.TestCase):
    def test_flat_plane_cartesian(self):
        x, y = sp.symbols('x y', real=True)
        dx, dy = sp.symbols('dx dy', real=True)
        ds2 = dx**2 + dy**2
        M = Spacetime((x, y), ds2)
        self.assertEqual(sp.simplify(M.Ricci), sp.zeros(2))
        self.assertEqual(sp.simplify(M.Ricci_scalar), 0)

    def test_flat_plane_polar(self):
        x, y = sp.symbols('x y', real=True)
        dx, dy = sp.symbols('dx dy', real=True)
        ds2 = dx**2 + dy**2
        M = Spacetime((x, y), ds2)

        r, phi = sp.symbols('r phi', real=True, positive=True)
        Mp = M.change_coordinates((r, phi), (r*sp.cos(phi), r*sp.sin(phi)))
        expected = sp.Matrix([[1, 0],[0, r**2]])
        self.assertEqual(sp.simplify(Mp.metric - expected), sp.zeros(2))
        self.assertEqual(sp.simplify(Mp.Ricci), sp.zeros(2))
        self.assertEqual(sp.simplify(Mp.Ricci_scalar), 0)

    def test_two_sphere(self):
        theta, phi = sp.symbols('theta phi', real=True)
        dtheta, dphi = sp.symbols('dtheta dphi', real=True)
        ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
        S2 = Spacetime((theta, phi), ds2)

        Gamma = S2.Gamma
        self.assertEqual(sp.simplify(Gamma[0,1,1] + sp.sin(theta)*sp.cos(theta)), 0)
        self.assertEqual(sp.simplify(Gamma[1,0,1] - sp.cot(theta)), 0)
        self.assertEqual(sp.simplify(Gamma[1,1,0] - sp.cot(theta)), 0)
        # With the first-fourth contraction, positive curvature picks up a minus sign.
        self.assertEqual(sp.simplify(S2.Ricci + S2.metric), sp.zeros(2))
        self.assertEqual(sp.simplify(S2.Ricci_scalar + 2), 0)

    def test_hyperbolic_plane(self):
        theta, phi = sp.symbols('theta phi', real=True)
        dtheta, dphi = sp.symbols('dtheta dphi', real=True)
        ds2 = dtheta**2 + sp.sinh(theta)**2 * dphi**2
        H2 = Spacetime((theta, phi), ds2)

        # Negative curvature space now yields positive Ricci eigenvalues.
        self.assertEqual(sp.simplify(H2.Ricci - H2.metric), sp.zeros(2))
        self.assertEqual(sp.simplify(H2.Ricci_scalar - 2), 0)

    def test_print_methods(self):
        theta, phi = sp.symbols('theta phi', real=True)
        dtheta, dphi = sp.symbols('dtheta dphi', real=True)
        ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
        S2 = Spacetime((theta, phi), ds2)

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S2.print_metric()
        out = buf.getvalue()
        assert "ds^2 =" in out and "dtheta**2" in out and "dphi**2" in out

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S2.print_nonzero()
        out = buf.getvalue()
        assert "Ricci scalar R" in out

class TestSpacetime4D(unittest.TestCase):
    def test_minkowski_4d_cartesian(self):
        t, x, y, z = sp.symbols('t x y z', real=True)
        dt, dx, dy, dz = sp.symbols('dt dx dy dz', real=True)
        ds2 = dt**2 - dx**2 - dy**2 - dz**2
        M4 = Spacetime((t, x, y, z), ds2)
        self.assertEqual(sp.simplify(M4.Ricci), sp.zeros(4))
        self.assertEqual(sp.simplify(M4.Ricci_scalar), 0)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        self.assertEqual(sp.simplify(M4.Riemann[i,j,k,l]), 0)

    def test_schwarzschild_vacuum_4d(self):
        t, r, th, ph = sp.symbols('t r theta phi', real=True, positive=True)
        dt, dr, dth, dph = sp.symbols('dt dr dtheta dphi', real=True)
        M = sp.symbols('M', positive=True)
        f = 1 - 2*M/r
        ds2 = f*dt**2 - (f**-1)*dr**2 - r**2*(dth**2 + sp.sin(th)**2 * dph**2)
        S = Spacetime((t, r, th, ph), ds2)

        self.assertEqual(sp.simplify(S.Ricci), sp.zeros(4))
        self.assertEqual(sp.simplify(S.Ricci_scalar), 0)

        found = False
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        if sp.simplify(S.Riemann[i,j,k,l]) != 0:
                            found = True
                            break
                    if found: break
                if found: break
            if found: break
        self.assertTrue(found)

        # Kretschmann scalar K = 48 M^2 / r^6
        g = S.metric
        ginv = S.inv_metric
        Riem = S.Riemann
        R_down = sp.MutableDenseNDimArray([0]*4**4, (4,4,4,4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        s = 0
                        for m in range(4):
                            s += g[i,m]*Riem[m,j,k,l]
                        R_down[i,j,k,l] = sp.simplify(s)
        K = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for a in range(4):
                            for b in range(4):
                                for c in range(4):
                                    for d in range(4):
                                        K += ginv[i,a]*ginv[j,b]*ginv[k,c]*ginv[l,d]*R_down[i,j,k,l]*R_down[a,b,c,d]
        K = sp.simplify(sp.together(sp.factor(K)))
        expected = sp.simplify(48*M**2 / r**6)
        self.assertEqual(sp.simplify(K - expected), 0)

    def test_print_selector_flags(self):
        t, x, y, z = sp.symbols('t x y z', real=True)
        dt, dx, dy, dz = sp.symbols('dt dx dy dz', real=True)
        ds2 = dt**2 - dx**2 - dy**2 - dz**2
        M4 = Spacetime((t, x, y, z), ds2)

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            M4.print_nonzero(show_christoffel=False, show_riemann=False, show_ricci=True, show_scalar=True)
        out = buf.getvalue()
        self.assertIn("Ricci tensor components", out)
        self.assertIn("Ricci scalar R", out)
        self.assertNotIn("Christoffel", out)
        self.assertNotIn("Riemann tensor", out)

class TestSpacetimeCurvatureScalars(unittest.TestCase):
    def test_desitter_static_patch(self):
        t, r, th, ph = sp.symbols('t r theta phi', real=True, positive=True)
        dt, dr, dth, dph = sp.symbols('dt dr dtheta dphi', real=True)
        H = sp.symbols('H', positive=True)

        f = 1 - H**2*r**2
        ds2 = -f*dt**2 + f**-1*dr**2 + r**2*(dth**2 + sp.sin(th)**2 * dph**2)
        dS = Spacetime((t, r, th, ph), ds2)

        # Constant curvature space: R_ij = -3 H^2 g_ij and scalar = -12 H^2.
        self.assertEqual(sp.simplify(dS.Ricci + 3*H**2*dS.metric), sp.zeros(4))
        self.assertEqual(sp.simplify(dS.Ricci_scalar + 12*H**2), 0)


class TestSpacetimeInternals(unittest.TestCase):
    def test_property_caching(self):
        x, y = sp.symbols('x y', real=True)
        dx, dy = sp.symbols('dx dy', real=True)
        ds2 = dx**2 + dy**2
        M = Spacetime((x, y), ds2)

        inv1 = M.inv_metric
        self.assertIs(inv1, M.inv_metric)

        Gamma1 = M.Gamma
        self.assertIs(Gamma1, M.Gamma)

        Riemann1 = M.Riemann
        self.assertIs(Riemann1, M.Riemann)

    def test_change_coordinates_mismatch_raises(self):
        x, y, r = sp.symbols('x y r', real=True)
        dx, dy = sp.symbols('dx dy', real=True)
        ds2 = dx**2 + dy**2
        M = Spacetime((x, y), ds2)

        with self.assertRaises(ValueError):
            M.change_coordinates((r,), (r,))

    def test_print_nonzero_latex_all_pairs(self):
        theta, phi = sp.symbols('theta phi', real=True)
        dtheta, dphi = sp.symbols('dtheta dphi', real=True)
        ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
        S2 = Spacetime((theta, phi), ds2)

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S2.print_nonzero(latex=True, show_all_pairs=True, show_riemann=False, show_ricci=False, show_scalar=False)
        out = buf.getvalue()
        self.assertIn("# Non-zero Christoffel symbols", out)
        self.assertIn(r"\Gamma^0_{11}", out)
        self.assertIn(r"\Gamma^1_{10}", out)
        self.assertNotIn("Ricci tensor components", out)

    def test_render_latex_pdf_invokes_pdflatex(self):
        theta, phi = sp.symbols('theta phi', real=True)
        dtheta, dphi = sp.symbols('dtheta dphi', real=True)
        ds2 = dtheta**2 + sp.sin(theta)**2 * dphi**2
        S2 = Spacetime((theta, phi), ds2)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "s2_report"
            completed = subprocess.CompletedProcess(args=["pdflatex"], returncode=0, stdout=b"", stderr=b"")
            with mock.patch("spacetime.spacetime.subprocess.run", return_value=completed) as mocked_run:
                pdf_path = S2.render_latex_pdf(target, show_riemann=False, cleanup_auxiliary=False)

            tex_path = target.with_suffix(".tex")
            self.assertEqual(pdf_path, target.with_suffix(".pdf").resolve())
            self.assertTrue(tex_path.exists())
            tex_contents = tex_path.read_text()
            self.assertIn("Spacetime Report", tex_contents)
            self.assertIn(r"d\theta", tex_contents)

            mocked_run.assert_called_once()
            args, kwargs = mocked_run.call_args
            self.assertIn("pdflatex", args[0][0])
            self.assertEqual(kwargs.get("cwd"), Path(tmpdir))
            self.assertIn(tex_path.name, args[0])

class TestSpacetimeOffDiagonal4D(unittest.TestCase):
    def test_rotating_minkowski_cylindrical(self):
        # Rotating frame with angular velocity Omega in cylindrical coords (t,r,phi,z)
        import sympy as sp
        from spacetime import Spacetime

        t, r, phi, z = sp.symbols('t r phi z', real=True)
        dt, dr, dphi, dz = sp.symbols('dt dr dphi dz', real=True)
        Omega = sp.symbols('Omega', real=True)

        # ds^2 = (1 - Omega^2 r^2) dt^2 + 2 Omega r^2 dt dphi - dr^2 - r^2 dphi^2 - dz^2
        ds2 = (1 - Omega**2*r**2)*dt**2 + 2*Omega*r**2*dt*dphi - dr**2 - r**2*dphi**2 - dz**2
        Mrot = Spacetime((t, r, phi, z), ds2)

        # It is still flat spacetime in a non-inertial chart: all curvature invariants vanish
        self.assertEqual(sp.simplify(Mrot.Ricci), sp.zeros(4))
        self.assertEqual(sp.simplify(Mrot.Ricci_scalar), 0)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        self.assertEqual(sp.simplify(Mrot.Riemann[i,j,k,l]), 0)

    def test_painleve_gullstrand_schwarzschild(self):
        # Painlevé–Gullstrand (PG) form of Schwarzschild has an off-diagonal dt dr term
        import sympy as sp
        from spacetime import Spacetime

        t, r, th, ph = sp.symbols('t r theta phi', real=True, positive=True)
        dt, dr, dth, dph = sp.symbols('dt dr dtheta dphi', real=True)
        M = sp.symbols('M', positive=True)

        # PG metric with signature (+,-,-,-):
        # ds^2 = (1 - 2M/r) dt^2 - 2 sqrt(2M/r) dt dr - dr^2 - r^2(dtheta^2 + sin^2theta dphi^2)
        f = 1 - 2*M/r
        v = sp.sqrt(2*M/r)
        ds2 = f*dt**2 - 2*v*dt*dr - dr**2 - r**2*(dth**2 + sp.sin(th)**2*dph**2)

        PG = Spacetime((t, r, th, ph), ds2)

        # Vacuum: Ricci=0, R=0 (same spacetime as Schwarzschild)
        self.assertEqual(sp.simplify(PG.Ricci), sp.zeros(4))
        self.assertEqual(sp.simplify(PG.Ricci_scalar), 0)

        # Riemann non-zero (curved)
        nonzero = False
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        if sp.simplify(PG.Riemann[i,j,k,l]) != 0:
                            nonzero = True
                            break
                if nonzero: break
            if nonzero: break
        self.assertTrue(nonzero)

        # Kretschmann scalar: K = 48 M^2 / r^6
        g = PG.metric
        ginv = PG.inv_metric
        Riem = PG.Riemann

        # Lower first index: R_{i j k l} = g_{i m} R^m_{ j k l}
        R_down = sp.MutableDenseNDimArray([0]*4**4, (4,4,4,4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        s = 0
                        for m in range(4):
                            s += g[i,m]*Riem[m,j,k,l]
                        R_down[i,j,k,l] = sp.simplify(s)

        K = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for a in range(4):
                            for b in range(4):
                                for c in range(4):
                                    for d in range(4):
                                        K += ginv[i,a]*ginv[j,b]*ginv[k,c]*ginv[l,d]*R_down[i,j,k,l]*R_down[a,b,c,d]
        K = sp.simplify(sp.together(sp.factor(K)))
        expected = sp.simplify(48*M**2 / r**6)
        self.assertEqual(sp.simplify(K - expected), 0)


if __name__ == "__main__":
    unittest.main()
