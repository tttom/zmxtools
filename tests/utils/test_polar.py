import unittest

import numpy.testing as npt
import numpy as np

from zmxtools.utils.polar import cart2pol, pol2cart


class TestPolar(unittest.TestCase):
    def test_cart2pol_scalar(self):
        npt.assert_array_equal(cart2pol(0, 1), (1, 0))
        npt.assert_array_equal(cart2pol(0, 2), (2, 0))
        npt.assert_array_equal(cart2pol(0, -3), (3, np.pi))
        npt.assert_array_equal(cart2pol(1, 0), (1, np.pi / 2))
        npt.assert_array_equal(cart2pol(-4, 0), (4, -np.pi / 2))
        npt.assert_array_equal(cart2pol(4, 3), (5, np.arctan2(4, 3)))
        npt.assert_array_equal(cart2pol(0, 0), (0, 0))

    def test_cart2pol_vector_broadcast(self):
        npt.assert_array_equal(
            cart2pol(0, [1, 2, -3]),
            ((1, 2, 3), (0, 0, np.pi)),
        )
        npt.assert_array_equal(
            cart2pol(0, [[1], [2], [-3]]),
            (([1], [2], [3]), ([0], [0], [np.pi])),
        )
        npt.assert_array_equal(
            cart2pol([1, 2, -3], 0),
            ((1, 2, 3), (np.pi/2, np.pi/2, -np.pi/2)),
        )
        npt.assert_array_equal(
            cart2pol([[1], [2], [-3]], [[[0]]]),
            ([[[1], [2], [3]]], [[[np.pi/2], [np.pi/2], [-np.pi/2]]]),
        )
        npt.assert_array_equal(
            cart2pol([[0], [1]], [1, 2, -3]),
            (
                [(1, 2, 3), (np.sqrt(2), np.sqrt(5), np.sqrt(10))],
                [(0, 0, np.pi), (np.pi/4, np.arctan2(1, 2), np.arctan2(1, -3))],
            ),
        )

    def test_pol2cart(self):
        npt.assert_array_equal(pol2cart(1, 0), (0, 1))
        npt.assert_array_equal(pol2cart(2, 0), (0, 2))
        npt.assert_almost_equal(pol2cart(3, np.pi), (0, -3))
        npt.assert_almost_equal(pol2cart(1, np.pi / 2), (1, 0))
        npt.assert_almost_equal(pol2cart(4, -np.pi / 2), (-4, 0))
        npt.assert_almost_equal(pol2cart(5, np.arctan2(4, 3)), (4, 3))
        npt.assert_array_equal(pol2cart(0, 0), (0, 0))


if __name__ == '__main__':
    unittest.main()
