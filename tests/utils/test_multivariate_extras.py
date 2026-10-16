import numpy as np
import pytest
from numpy import testing as npt

from zmxtools.utils.multivariate import PolarPolynomial, Polynomial


class TestPolynomialExtras:
    def test_exponents_padding(self):
        p = Polynomial([[1, 2], [3, 4]], "xy")
        p.exponents = [[0]]  # too short: padded with range
        npt.assert_array_equal(p.exponents[0], [0, 1])

    def test_exponents_too_long(self):
        p = Polynomial([1, 2], "x")
        with pytest.raises(ValueError):
            p.exponents = [[0, 1, 2]]

    def test_rsub(self):
        p = Polynomial([2, 3], "x")
        q = 1 - p
        self.check_eq(q, Polynomial([1 - 2, -3], "x"))

    def test_mul_scalar(self):
        p = Polynomial([2, 3], "x")
        self.check_eq(p * 2, Polynomial([4, 6], "x"))
        self.check_eq(2 * p, Polynomial([4, 6], "x"))

    def test_mul_polynomial_not_implemented(self):
        p = Polynomial([2, 3], "x")
        with pytest.raises(NotImplementedError):
            p * Polynomial([1], "x")

    def test_str_complex(self):
        p = Polynomial([1, 1, 1, 1, 1], "x")
        p.coefficients = np.asarray([1j, -1j, 1 + 1j, 0, -1])
        s = str(p)
        npt.assert_equal("i" in s, True)
        q = Polynomial([0], "x")
        npt.assert_equal(str(q), "0.0")

    def test_eq_other(self):
        p = Polynomial([2, 3], "x")
        npt.assert_equal(p == 3, False)
        npt.assert_equal(hash(p) == hash(Polynomial([2, 3], "x")), True)
        npt.assert_equal(repr(p).startswith("Polynomial("), True)

    def test_call_kwargs(self):
        p = Polynomial([2, 5], "x")
        npt.assert_allclose(p(x=2), 12)

    def test_grad(self):
        p = Polynomial([[1, 2], [3, 4]], "xy")
        g = p.grad()
        npt.assert_equal(len(g), 2)

    def check_eq(self, p, q):
        npt.assert_array_equal(np.asarray(p.coefficients), np.asarray(q.coefficients))


class TestPolarPolynomialExtras:
    def setup_method(self):
        self.p = PolarPolynomial([[1, 2, 3], [4, 5, 6]], labels=("rho", "phi"))

    def test_call(self):
        v = self.p(0.5, 0.25)
        npt.assert_equal(np.isfinite(v), True)
        v2 = self.p(rho=0.5, phi=0.25)
        npt.assert_allclose(v, v2)

    def test_cartesian(self):
        v = self.p.cartesian(0.3, 0.4)
        npt.assert_allclose(v, self.p(0.5, np.arctan2(0.3, 0.4)))

    def test_cartesian_grad(self):
        g = self.p.cartesian_grad(0.3, 0.4)
        npt.assert_equal(g.shape[-1], 2)
        g0 = self.p.cartesian_grad(0.0, 0.0)
        npt.assert_equal(np.all(np.isfinite(g0)), True)

    def test_repr(self):
        npt.assert_equal(repr(self.p).startswith("PolarPolynomial("), True)
