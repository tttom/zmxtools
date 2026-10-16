import unittest

import numpy as np
from numpy import testing as npt

from zmxtools.utils.zernike import (
    Fit,
    Polynomial,
    PolynomialBasis,
    fringe2index,
    fringe2orders,
    index2fringe,
    index2noll,
    index2orders,
    noll2index,
    noll2orders,
    orders2fringe,
    orders2index,
    orders2noll,
)


class TestZernikeExtras(unittest.TestCase):
    def test_polar_gradient(self):
        b = PolynomialBasis(n=2, m=0)
        g = b.polar_gradient(0.5, 0.25)
        npt.assert_equal(np.all(np.isfinite(g)), True)
        b_odd = PolynomialBasis(n=3, m=1)
        g_odd = b_odd.polar_gradient(0.5, 0.25)
        npt.assert_equal(np.all(np.isfinite(g_odd)), True)

    def test_call_cartesian(self):
        p = Polynomial([0, 0, 1], indices=[0, 1, 2])
        v = p(y=0.3, x=0.4)
        npt.assert_allclose(v, p(rho=0.5, phi=np.arctan2(0.3, 0.4)))

    def test_call_complex(self):
        p = Polynomial([0, 0, 1], indices=[0, 1, 2])
        v = p(0.4 + 0.3j)
        npt.assert_allclose(v, p(y=0.3, x=0.4))

    def test_basis_names(self):
        names = {
            (0, 0): "piston",
            (1, -1): "tilt",
            (1, 1): "tip",
            (2, 0): "defocus",
            (2, -2): "oblique",
            (2, 2): "vertical",
            (3, -1): "vertical",
            (3, 1): "horizontal",
            (4, 0): "primary",
            (3, 3): "oblique",
            (4, 2): "vertical",
            (4, -2): "oblique",
            (5, 3): "horizontal",
        }
        for (n, m), expected in names.items():
            with self.subTest(n=n, m=m):
                npt.assert_equal(str(PolynomialBasis(n=n, m=m).name).startswith(expected), True)

    def test_polynomial_arithmetic(self):
        p = Polynomial([1, 2, 3])
        q = p + 1
        npt.assert_array_equal(q.coefficients, [2, 2, 3])
        r = 1 + p
        npt.assert_array_equal(r.coefficients, [2, 2, 3])
        neg = -p
        npt.assert_array_equal(neg.coefficients, [-1, -2, -3])
        s = p - 1
        npt.assert_array_equal(s.coefficients, [0, 2, 3])
        t = 1 - p
        npt.assert_array_equal(t.coefficients, [0, -2, -3])
        m = p * 2
        npt.assert_array_equal(m.coefficients, [2, 4, 6])
        m2 = 2 * p
        npt.assert_array_equal(m2.coefficients, [2, 4, 6])
        d = p / 2
        npt.assert_array_equal(d.coefficients, [0.5, 1, 1.5])
        npt.assert_array_equal(p.coefficients, [1, 2, 3])  # division does not mutate
        p2 = Polynomial([4, 4, 4])
        p2 /= 2
        npt.assert_array_equal(p2.coefficients, [2, 2, 2])
        npt.assert_equal(str(p).startswith("Z"), True)
        npt.assert_equal(repr(p).startswith("Polynomial("), True)

    def test_polynomial_str(self):
        p = Polynomial([1, 2, 3])
        npt.assert_equal("Z" in str(p), True)

    def test_index_conversions_roundtrip(self):
        idx = np.arange(1, 11)
        nm = np.array(index2orders(idx))
        npt.assert_array_equal(orders2index(nm[0], nm[1]), idx)
        npt.assert_array_equal(noll2index(index2noll(idx)), idx)
        npt.assert_array_equal(fringe2index(index2fringe(idx)), idx)
        npt.assert_array_equal(orders2noll(*noll2orders(idx)), idx)
        npt.assert_array_equal(orders2fringe(*fringe2orders(idx)), idx)

    def test_fit_covariant(self):
        rho = np.linspace(0, 1, 5)
        phi = np.linspace(0, 2 * np.pi, 7)
        rho2d, phi2d = np.meshgrid(rho, phi, indexing="ij")
        z = 1 + 0.5 * rho2d * np.cos(phi2d)
        fit = Fit(rho=rho2d, phi=phi2d, z=z, order=4)
        npt.assert_equal(fit.coefficients.shape[-1], 4)
        npt.assert_equal(fit.covariant.shape[-1], 4)
        npt.assert_equal(np.all(np.isfinite(fit.error)), True)
        npt.assert_equal(str(fit) != "", True)
        npt.assert_equal(repr(fit) != "", True)


if __name__ == "__main__":
    unittest.main()
