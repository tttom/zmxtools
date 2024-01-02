import unittest
import numpy.testing as npt

from zmxtools.utils import to_length

import numpy as np


class TestPadToLength(unittest.TestCase):
    def setUp(self):
        pass

    def test_pad_to_length(self):
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 5), np.array([1, 2, 3, 4, 0]),
                               err_msg="Could not extend array by one.")
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 4), np.array([1, 2, 3, 4]),
                               err_msg="Could not extend array by zero.")
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 5, -1), np.array([1, 2, 3, 4, -1]),
                               err_msg="Could not extend array by one with value -1.")
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 6), np.array([1, 2, 3, 4, 0, 0]),
                               err_msg="Could not extend array by two.")
        npt.assert_array_equal(to_length([1, 2, 3, 4], 6), np.array([1, 2, 3, 4, 0, 0]),
                               err_msg="Could not extend list by two.")
        npt.assert_array_equal(to_length((1, 2, 3, 4), 6), np.array([1, 2, 3, 4, 0, 0]),
                               err_msg="Could not extend tuple by two.")
        npt.assert_array_equal(to_length(10, 4), np.array([10, 0, 0, 0]),
                               err_msg="Could not extend scalar by 3.")

    def test_crop_to_length(self):
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 3), np.array([1, 2, 3]),
                               err_msg="Could not crop array by one.")
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 3, -1), np.array([1, 2, 3]),
                               err_msg="Could not crop array by one with pad value -1.")
        npt.assert_array_equal(to_length(np.array([1, 2, 3, 4]), 1), np.array([1]),
                               err_msg="Could not crop array by length one.")
        npt.assert_array_equal(to_length([1, 2, 3, 4], 0), np.array([], dtype=int),
                               err_msg="Could not crop array to length zero.")
        npt.assert_array_equal(to_length((1, 2, 3, 4), 2), np.array([1, 2]),
                               err_msg="Could not crop tuple by two.")
        npt.assert_array_equal(to_length(10, 1), np.array([10]),
                               err_msg="Could not crop scalar to same length.")
        npt.assert_array_equal(to_length(10, 0), np.array([], dtype=int),
                               err_msg="Could not crop scalar to zero.")


if __name__ == '__main__':
    unittest.main()
