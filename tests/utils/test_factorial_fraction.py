import unittest
import numpy.testing as npt

from zmxtools.utils import factorial_fraction

import numpy as np


class TestFactorialFraction(unittest.TestCase):
    def setUp(self):
        self.f0 = factorial_fraction(0)
        self.f1 = factorial_fraction(1)
        self.f4 = factorial_fraction(4)
        self.f6 = factorial_fraction(6, 1)
        self.i0 = factorial_fraction(1, 0)
        self.i1 = factorial_fraction(1, 1)
        self.i4 = factorial_fraction(denominator=4)
        self.i6 = factorial_fraction(1, 6)

    def test_natural(self):
        npt.assert_equal(self.f0, 1, '0! not correct')
        npt.assert_equal(self.f1, 1, '1! not correct')
        npt.assert_equal(self.f4, 4*3*2*1, '4! not correct')
        npt.assert_equal(self.f6, 6*5*4*3*2*1, '6! not correct')

    def test_inverse_natural(self):
        npt.assert_equal(self.i0, 1, '1/0! not correct')
        npt.assert_equal(self.i1, 1, '1/1! not correct')
        npt.assert_equal(self.i4, 1/(4*3*2*1), '1/4! not correct')
        npt.assert_almost_equal(self.i6, 1/(6*5*4*3*2*1), 12, '1/6! not correct')

    def test_fraction_above_one(self):
        npt.assert_equal(factorial_fraction(6, 4), 6*5, '6!/4! not correct')

    def test_fraction_below_one(self):
        npt.assert_equal(factorial_fraction(4, 6), 1/(6*5), '4!/6! not correct')

    def test_fraction_array_natural(self):
        npt.assert_equal(factorial_fraction((1, 2, 3, 4)), np.array((1, 2, 6, 24)),
                         'tuple with natural fractions not correct')
        npt.assert_equal(factorial_fraction([1, 2, 3, 4]), np.array((1, 2, 6, 24)),
                         'list with natural fractions not correct')
        npt.assert_equal(factorial_fraction(np.array([1, 2, 3, 4])), np.array((1, 2, 6, 24)),
                         'vector with natural fractions not correct')
        npt.assert_equal(factorial_fraction(np.array([[1, 2, 3, 4]])), np.array([[1, 2, 6, 24]]),
                         'vector along dim 1 with natural fractions not correct')
        npt.assert_equal(factorial_fraction(np.array([[1, 2], [3, 4]])), np.array([[1, 2], [6, 24]]),
                         'array with natural fractions not correct')

    def test_fraction_array_inverse_natural(self):
        npt.assert_equal(factorial_fraction(0, (1, 2, 3, 4)), np.array((1, 1/2, 1/6, 1/24)),
                         'tuple with natural fractions not correct')
        npt.assert_equal(factorial_fraction(0, [1, 2, 3, 4]), np.array((1, 1/2, 1/6, 1/24)),
                         'list with natural fractions not correct')
        npt.assert_equal(factorial_fraction(0, np.array([1, 2, 3, 4])), np.array((1, 1/2, 1/6, 1/24)),
                         'vector with natural fractions not correct')
        npt.assert_equal(factorial_fraction(0, np.array([[1, 2, 3, 4]])), np.array([[1, 1/2, 1/6, 1/24]]),
                         'vector along dim 1 with natural fractions not correct')
        npt.assert_equal(factorial_fraction(0, np.array([[1, 2], [3, 4]])), np.array([[1, 1/2], [1/6, 1/24]]),
                         'array with natural fractions not correct')
        npt.assert_equal(factorial_fraction(1, (1, 2, 3, 4)), np.array((1, 1/2, 1/6, 1/24)),
                         'tuple with natural fractions not correct')
        npt.assert_equal(factorial_fraction(1, [1, 2, 3, 4]), np.array((1, 1/2, 1/6, 1/24)),
                         'list with natural fractions not correct')
        npt.assert_equal(factorial_fraction(1, np.array([1, 2, 3, 4])), np.array((1, 1/2, 1/6, 1/24)),
                         'vector with natural fractions not correct')
        npt.assert_equal(factorial_fraction(1, np.array([[1, 2, 3, 4]])), np.array([[1, 1/2, 1/6, 1/24]]),
                         'vector along dim 1 with natural fractions not correct')
        npt.assert_equal(factorial_fraction(1, np.array([[1, 2], [3, 4]])), np.array([[1, 1/2], [1/6, 1/24]]),
                         'array with natural fractions not correct')

    def test_fraction_array(self):
        npt.assert_equal(factorial_fraction(2, (1, 2, 3, 4)), np.array((2, 1, 1/3, 1/12)),
                         'tuple with natural fractions not correct')
        npt.assert_equal(factorial_fraction(2, [1, 2, 3, 4]), np.array((2, 1, 1/3, 1/12)),
                         'list with natural fractions not correct')
        npt.assert_equal(factorial_fraction(2, np.array([1, 2, 3, 4])), np.array((2, 1, 1/3, 1/12)),
                         'vector with natural fractions not correct')
        npt.assert_equal(factorial_fraction(2, np.array([[1, 2, 3, 4]])), np.array([[2, 1, 1/3, 1/12]]),
                         'vector along dim 1 with natural fractions not correct')
        npt.assert_equal(factorial_fraction(2, np.array([[1, 2], [3, 4]])), np.array([[2, 1], [1/3, 1/12]]),
                         'array with natural fractions not correct')

    def test_fraction_arrays(self):
        npt.assert_equal(factorial_fraction((2, 2, 2, 2), (1, 2, 3, 4)), np.array((2, 1, 1/3, 1/12)),
                         'tuple with natural fractions not correct')
        npt.assert_equal(factorial_fraction((2, 2, 2, 2), [1, 2, 3, 4]), np.array((2, 1, 1/3, 1/12)),
                         'list with natural fractions not correct')
        npt.assert_equal(factorial_fraction((2, 2, 2, 2), np.array([1, 2, 3, 4])), np.array((2, 1, 1/3, 1/12)),
                         'vector with natural fractions not correct')
        npt.assert_equal(factorial_fraction((2, 2, 2, 2), np.array([[1, 2, 3, 4]])), np.array([[2, 1, 1/3, 1/12]]),
                         'vector along dim 1 with natural fractions not correct')
        npt.assert_equal(factorial_fraction([[2, 2], [2, 2]], np.array([[1, 2], [3, 4]])), np.array([[2, 1], [1/3, 1/12]]),
                         'array with natural fractions not correct')
        npt.assert_equal(factorial_fraction((2, 2, 3, 3), [1, 2, 3, 4]), np.array((2, 1, 1, 1/4)),
                         'list with natural fractions not correct')


if __name__ == '__main__':
    unittest.main()
