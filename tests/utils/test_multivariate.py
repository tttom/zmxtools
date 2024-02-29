import unittest
import numpy.testing as npt

from zmxtools.utils.multivariate import Polynomial
from zmxtools.utils.array import array_type

import numpy as np
from tests.utils import log


class TestPolynomial(unittest.TestCase):
    def check_eq(self, p0, p1):
        npt.assert_equal(p0 == p1, True, err_msg=f"{repr(p0)} == {p0} != {p1} == {repr(p1)}")
        npt.assert_equal(p1 == p0, True, err_msg=f"{repr(p0)} == {p0} != {p1} == {repr(p1)}")

    def check_neq(self, p0, p1):
        npt.assert_equal(p0 != p1, True, err_msg=f"{repr(p0)} == {p0} == {p1} == {repr(p1)}")
        npt.assert_equal(p1 != p0, True, err_msg=f"{repr(p0)} == {p0} == {p1} == {repr(p1)}")

    def test_init(self):
        p = Polynomial([1, -4], "x", [[0, 1]])
        npt.assert_equal(p.ndim, 1)
        npt.assert_equal(p.shape, (2, ))
        npt.assert_equal(p.labels, ("x",))
        npt.assert_equal(p.coefficients, (1, -4))
        npt.assert_equal(p.exponents, ((0, 1), ))
        p = Polynomial([-4], "x", [[1]])
        npt.assert_equal(p.ndim, 1)
        npt.assert_equal(p.shape, (1, ))
        npt.assert_equal(p.labels, ("x",))
        npt.assert_equal(p.coefficients, (-4))
        npt.assert_equal(p.exponents, ((1, ), ))
        p = Polynomial([-4], "x", [[0]])
        npt.assert_equal(p.ndim, 1)
        npt.assert_equal(p.shape, (1, ))
        npt.assert_equal(p.labels, ("x",))
        npt.assert_equal(p.coefficients, (-4))
        npt.assert_equal(p.exponents, ((0, ), ))

        log.debug("Testing 0 polynomial...")
        p = Polynomial([], "x", [[]])
        npt.assert_equal(p.ndim, 1)
        npt.assert_equal(p.shape, [0])
        npt.assert_equal(p.labels, ("x",))
        npt.assert_equal(p.coefficients, (-4))
        npt.assert_equal(p.exponents, ([], ))

        p = Polynomial([1, 0, -4], "y", [[2, 3, -5]])
        npt.assert_equal(p.ndim, 1)
        npt.assert_equal(p.shape, (3, ))
        npt.assert_equal(p.labels, ("y",))
        npt.assert_equal(p.coefficients, (1, 0, -4))
        npt.assert_equal(p.exponents, ((2, 3, -5), ))
        p = Polynomial([[1, 0, -4], [2, 4, 3]], "xy", [[0, 2], [2, 3, -5]])
        npt.assert_equal(p.ndim, 2)
        npt.assert_equal(p.shape, (2, 3))
        npt.assert_equal(p.labels, ("x", "y"))
        npt.assert_equal(p.coefficients, ((1, 0, -4), (2, 4, 3)))
        npt.assert_equal(p.exponents, ((0, 2), (2, 3, -5)))
        p = Polynomial([[1, 0, -4], [2, 4, 3]], ("apple", "orange"), [[0, 2], [2, 3, -5]])
        npt.assert_equal(p.ndim, 2)
        npt.assert_equal(p.shape, (2, 3))
        npt.assert_equal(p.labels, ["apple", "orange"])
        npt.assert_equal(p.coefficients, ((1, 0, -4), (2, 4, 3)))
        npt.assert_equal(p.exponents, ((0, 2), (2, 3, -5)))

        log.info("Testing default initializations...")
        p = Polynomial([[1, 0, -4], [2, 4, 3]], labels="xy", exponents=[[0, 2], ])
        npt.assert_equal(p.ndim, 2)
        npt.assert_equal(p.shape, (2, 3))
        npt.assert_equal(p.labels, ("x", "y"))
        npt.assert_equal(p.coefficients, ((1, 0, -4), (2, 4, 3)))
        npt.assert_equal(p.exponents, ((0, 2), (0, 1, 2)))
        p = Polynomial([[(1, 2), (0, 1), (-4, -5)], [(2, 3), (4, 5), (3, 0)]])
        npt.assert_equal(p.ndim, 3)
        npt.assert_equal(p.shape, (2, 3, 2))
        npt.assert_equal(p.labels, ("x₀", "x₁", "x₂"))
        npt.assert_equal(p.coefficients, (([1, 2], [0, 1], [-4, -5]), ([2, 3], [4, 5], [3, 0])))
        npt.assert_equal(p.exponents, ((0, 1), (0, 1, 2), (0, 1)))

    def test_str(self):
        p = Polynomial([2], "x")
        npt.assert_equal(str(p), "2.0")
        p = Polynomial([0], "x")
        npt.assert_equal(str(p), "0.0")
        p = Polynomial([], "x")
        npt.assert_equal(str(p), "0.0")
        p = Polynomial([2, 3], "x")
        npt.assert_equal(str(p), "2.0 + 3.0x")
        p = Polynomial([2, 3], "x", [[0, -1]])
        npt.assert_equal(str(p), "2.0 + 3.0x⁻¹")
        p = Polynomial([3], "x", [[-1]])
        npt.assert_equal(str(p), "3.0x⁻¹")
        p = Polynomial([2, 3, 4], "x")
        npt.assert_equal(str(p), "2.0 + 3.0x + 4.0x²")
        p = Polynomial([2, 3, 4, 5, 6], "x")
        npt.assert_equal(str(p), "2.0 + 3.0x + 4.0x² + 5.0x³ + 6.0x⁴")
        p = Polynomial([2, 3, 1, 5, 6], "x")
        npt.assert_equal(str(p), "2.0 + 3.0x + x² + 5.0x³ + 6.0x⁴")
        p = Polynomial([2, -3, 1, 5, 6], "x")
        npt.assert_equal(str(p), "2.0 - 3.0x + x² + 5.0x³ + 6.0x⁴")
        p = Polynomial([2, 3, -1, 5, 6], "x")
        npt.assert_equal(str(p), "2.0 + 3.0x - x² + 5.0x³ + 6.0x⁴")
        p = Polynomial([1, 3, -1, 5, 6], "x")
        npt.assert_equal(str(p), "1.0 + 3.0x - x² + 5.0x³ + 6.0x⁴")
        p = Polynomial([-1, 3, -1, 5, 6], "x")
        npt.assert_equal(str(p), "- 1.0 + 3.0x - x² + 5.0x³ + 6.0x⁴")
        p = Polynomial([[-1, 3, -1], [4, -5, 6]], "yx")
        npt.assert_equal(str(p), "- 1.0 + 3.0x - x² + 4.0y - 5.0yx + 6.0yx²")
        p = Polynomial([(-1, 4), (3, -5), (-1, 6)], "xy")
        npt.assert_equal(str(p), "- 1.0 + 4.0y + 3.0x - 5.0xy - x² + 6.0x²y")
        p = Polynomial([(-1, 4), (3, -5), (-1, 6)])
        npt.assert_equal(str(p), "- 1.0 + 4.0x₁ + 3.0x₀ - 5.0x₀x₁ - x₀² + 6.0x₀²x₁")

    def test_eq(self):
        p0 = Polynomial([2], "x")
        p1 = Polynomial([2], "x")
        self.check_eq(p0, p1)
        p0 = Polynomial([2], "x")
        p1 = Polynomial([2], "y")
        self.check_neq(p0, p1)
        p0 = Polynomial([2], "x")
        p1 = Polynomial([-2], "x")
        self.check_neq(p0, p1)
        p0 = Polynomial([2, 5], "x")
        p1 = Polynomial([2, 5], "x")
        self.check_eq(p0, p1)
        p0 = Polynomial([2, 5], "x")
        p1 = Polynomial([2, 4], "x")
        self.check_neq(p0, p1)
        p0 = Polynomial([2, 5], "x")
        p1 = Polynomial([2], "x")
        self.check_neq(p0, p1)

    def test_add_sub(self):
        p0 = Polynomial([2, 7], "x")
        p1 = Polynomial([4, -1], "x")
        p2 = Polynomial([4], "x")
        p3 = Polynomial([4, -1], "y")
        self.check_eq(p0 + 3.0, Polynomial([2 + 3.0, 7], "x"))
        self.check_eq(p0 - 3.0, Polynomial([2 - 3.0, 7], "x"))
        self.check_eq(3.0 + p0, Polynomial([3.0 + 2, 7], "x"))
        self.check_eq(-3.0 + p0, Polynomial([-3.0 + 2, 7], "x"))
        self.check_eq(p0 + p1, Polynomial([2 + 4, 7 - 1], "x"))
        self.check_eq(p0 + p2, Polynomial([2 + 4, 7], "x"))
        self.check_eq(p0 + p3, Polynomial([[2 + 4, -1], [7, 0]], "xy"))
        self.check_eq(p0 - p3, Polynomial([[2 - 4, 1], [7, 0]], "xy"))

        p0 += 3
        self.check_eq(p0, Polynomial([2 + 3.0, 7], "x"))
        p0 += p1
        self.check_eq(p0, Polynomial([2 + 3.0 + 4, 7 - 1], "x"))

    def test_mul_div(self):
        p0 = Polynomial([2, 7], "x")
        p1 = Polynomial([4, -1], "x")
        p2 = Polynomial([4], "x")
        p3 = Polynomial([4, -1], "y")
        self.check_eq(-p0, Polynomial([-2, -7], "x"))
        self.check_eq(p0 * 2, Polynomial([2 * 2, 7 * 2], "x"))
        self.check_eq(2 * p0, Polynomial([2 * 2, 2 * 7], "x"))
        self.check_eq(-2 * p0, Polynomial([-2 * 2, -2 * 7], "x"))
        self.check_eq(p0 / 2, Polynomial([2 / 2, 7 / 2], "x"))
        # Not yet implemented:
        # self.check_eq(p0 * p2, Polynomial([2 * 4, 7 * 4], "x"))
        # self.check_eq(p0 * p1, Polynomial([2 * 4, 7 * 4 - 2, -7], "x"))
        # self.check_eq(p0 * p3, Polynomial([[2 * 4, -2], [7 * 4, -7]], "xy"))

        p0 *= 0.5
        self.check_eq(p0, Polynomial([2 / 2, 7 / 2], "x"))

    def test_evaluation(self):
        def compare_array(a: array_type, b: array_type):
            npt.assert_equal(a.ndim, np.asarray(b).ndim, err_msg=f"ndim of arrays {a} != {b}")
            npt.assert_equal(a.shape, np.asarray(b).shape, err_msg=f"shape of arrays {a} != {b}")
            npt.assert_equal(a, b, err_msg=f"Arrays {a} != {b}")

        p = Polynomial([2, 5], "x")
        compare_array(p(3.0), 2 + 5 * 3.0)
        compare_array(p(x=3.0), 2 + 5 * 3.0)
        compare_array(p(x=-2), 2 + 5 * -2)

        def try_wrong_symbol():
            return p(y=3.0)
        npt.assert_raises(Exception, try_wrong_symbol)

        p = Polynomial([2, 5], "x")
        compare_array(p([3.0]), [2 + 5 * 3.0])
        compare_array(p([[3.0]]), [[2 + 5 * 3.0]])
        compare_array(p([3.0, -2]), [2 + 5 * 3.0, 2 + 5 * -2])
        compare_array(p([[3.0], [-2]]), [[2 + 3.0 * 5], [2 + -2 * 5]])

        p = Polynomial([(2, 0), (5, -1)], "xy")  # 2 + 5x - xy
        compare_array(p(3.0, 2.0), 2 + 5 * 3.0 - 3.0 * 2.0)
        compare_array(p(x=3.0, y=2.0), 2 + 5 * 3.0 - 3.0 * 2.0)
        compare_array(p(y=2.0, x=3.0), 2 + 5 * 3.0 - 3.0 * 2.0)
        compare_array(p(y=2.0, x=3.0), 2 + 5 * 3.0 - 3.0 * 2.0)
        compare_array(p([3.0], [2.0]), [2 + 5 * 3.0 - 3.0 * 2.0])
        compare_array(p([3.0], 2.0), [2 + 5 * 3.0 - 3.0 * 2.0])
        compare_array(p([3.0, 6.0], 2.0), [2 + 5 * 3.0 - 3.0 * 2.0, 2 + 5 * 6.0 - 6.0 * 2.0])
        compare_array(p([3.0, 6.0], [2.0]), [2 + 5 * 3.0 - 3.0 * 2.0, 2 + 5 * 6.0 - 6.0 * 2.0])
        compare_array(p([3.0, 6.0], [2.0, 1.0]), [2 + 5 * 3.0 - 3.0 * 2.0, 2 + 5 * 6.0 - 6.0 * 1.0])
        compare_array(p(3.0, [2.0]), [2 + 5 * 3.0 - 3.0 * 2.0])
        compare_array(p([[3.0], [6.0]], [2.0, -1.0]), [[2 + 5 * 3.0 - 3.0 * 2.0, 2 + 5 * 3.0 + 3.0],
                                                       [2 + 5 * 6.0 - 6.0 * 2.0, 2 + 5 * 6.0 + 6.0]])

    def test_gradient(self):
        p = Polynomial([2, 5, 3], "x")
        g = p.grad()
        npt.assert_equal(len(g), 1)
        self.check_eq(g[0], Polynomial([5, 3 * 2], "x"))

        p = Polynomial([(2, 6), (5, 0), (3, -1)], "xy")  # 2 + 6y + 5x + 3x² - x²y
        g = p.grad()
        npt.assert_equal(len(g), 2)
        self.check_eq(g[0], Polynomial([[5, 0], [6, -2]], "xy"))
        self.check_eq(g[1], Polynomial([[6], [0], [-1]], "xy"))

        p = Polynomial([(2, 6, 2), (5, 0, 3), (3, -1, -2)], "xy")  # 2 + 6y + 2y² + 5x + 3xy² + 3x² - x²y -2x²y²
        g = p.grad()
        npt.assert_equal(len(g), 2)
        self.check_eq(g[0], Polynomial([[5, 0, 3], [6, -2, -4]], "xy"))
        self.check_eq(g[1], Polynomial([[6, 4], [0, 6], [-1, -4]], "xy"))


if __name__ == "__main__":
    unittest.main()
