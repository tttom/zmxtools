import unittest

import numpy.testing as npt
import numpy as np

from zmxtools.optical_design.geometry import (EulerRotation, IDENTITY, Scaling, Translation,
                                              SphericalTransform, InverseSphericalTransform)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.origin = (0, 0, 0)

        self.vx = np.array([1, 0, 0])
        self.vy = np.array([0, 1, 0])
        self.vz = np.array([0, 0, 1])
        self.vd = np.array([1, 1, 1])
        self.px = np.array([1, 0, 0])
        self.py = np.array([0, 1, 0])
        self.pz = np.array([0, 0, 1])
        self.pd = np.array([1, 1, 1])

        self.hvx = np.array([1, 0, 0, 0])
        self.hvy = np.array([0, 1, 0, 0])
        self.hvz = np.array([0, 0, 1, 0])
        self.hvd = np.array([1, 1, 1, 0])
        self.hpx = np.array([1, 0, 0, 1])
        self.hpy = np.array([0, 1, 0, 1])
        self.hpz = np.array([0, 0, 1, 1])
        self.hpd = np.array([1, 1, 1, 1])

    def test_scale(self):
        shrink = Scaling(0.5)
        expand = Scaling(2)
        irregular = Scaling([0.5, 2, 1])

        npt.assert_array_equal(shrink.vector(self.origin, self.vd), (0.5, 0.5, 0.5))
        npt.assert_array_equal(shrink.point(self.pd), (0.5, 0.5, 0.5))
        npt.assert_array_equal(expand.vector(self.origin, self.vd), (2, 2, 2))
        npt.assert_array_equal(expand.point(self.pd), (2, 2, 2))

    def test_euler_rotation(self):
        id = EulerRotation([0, 0], [0, 1])


if __name__ == '__main__':
    unittest.main()
