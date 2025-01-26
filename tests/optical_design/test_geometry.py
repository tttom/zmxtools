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

        self.hvx = np.array([0, 1, 0, 0])
        self.hvy = np.array([0, 0, 1, 0])
        self.hvz = np.array([0, 0, 0, 1])
        self.hvd = np.array([0, 1, 1, 1])
        self.hpx = np.array([1, 1, 0, 0])
        self.hpy = np.array([1, 0, 1, 0])
        self.hpz = np.array([1, 0, 0, 1])
        self.hpd = np.array([1, 1, 1, 1])

    def test_identity(self):
        identity = IDENTITY

        npt.assert_array_equal(identity.matrix, np.eye(4))
        npt.assert_array_equal(identity.scale, np.ones(3))
        npt.assert_equal(identity.inv, identity)

        npt.assert_array_equal(identity.homogeneous(self.hvx), self.hvx)
        npt.assert_array_equal(identity.vector(self.vx), self.vx)
        npt.assert_array_equal(identity.point(self.px), self.px)

        npt.assert_array_equal(identity.homogeneous(self.hvd), self.hvd)
        npt.assert_array_equal(identity.vector(self.vd), self.vd)
        npt.assert_array_equal(identity.point(self.pd), self.pd)

        npt.assert_array_equal(identity.homogeneous([self.hpx, self.hvd, self.hpz]),
                               [self.hpx, self.hvd, self.hpz],
                               )
        npt.assert_array_equal(identity.vector([self.vx, self.vd]), [self.vx, self.vd])
        npt.assert_array_equal(identity.point([self.px, self.pd]), [self.px, self.pd])

        npt.assert_array_equal(identity.homogeneous([[self.hpx, self.hvd, self.hpz]]),
                               np.asarray([[self.hpx, self.hvd, self.hpz]]),
                               )
        npt.assert_array_equal(identity.vector([[self.vx, self.vd]]), np.asarray([[self.vx, self.vd]]))
        npt.assert_array_equal(identity.point([[self.px, self.pd]]), np.asarray([[self.px, self.pd]]))

        npt.assert_array_equal(identity.homogeneous([[self.hpx], [self.hvd], [self.hpz]]),
                               np.asarray([[self.hpx], [self.hvd], [self.hpz]]),
                               )
        npt.assert_array_equal(identity.vector([[self.vx], [self.vd]]), np.asarray([[self.vx], [self.vd]]))
        npt.assert_array_equal(identity.point([[self.px], [self.pd]]), np.asarray([[self.px], [self.pd]]))

        npt.assert_equal(identity @ identity == identity, True)

    def test_scale(self):
        shrink = Scaling(0.5)
        expand = Scaling(2)
        irregular = Scaling([0.5, 2, 1])

        npt.assert_array_equal(shrink.scale, [0.5, 0.5, 0.5])
        npt.assert_array_equal(expand.scale, [2, 2, 2])
        npt.assert_array_equal(irregular.scale, [0.5, 2, 1])

        npt.assert_array_equal(shrink.matrix, np.diag([1, 0.5, 0.5, 0.5]))
        npt.assert_array_equal(expand.matrix, np.diag([1, 2, 2, 2]))
        npt.assert_array_equal(irregular.matrix, np.diag([1, 0.5, 2, 1]))

        npt.assert_array_equal(shrink.vector(self.vd), (0.5, 0.5, 0.5))
        npt.assert_array_equal(shrink.point(self.pd), (0.5, 0.5, 0.5))
        npt.assert_array_equal(shrink.homogeneous(self.hpd), (1, 0.5, 0.5, 0.5))
        npt.assert_array_equal(expand.vector(self.vd), (2, 2, 2))
        npt.assert_array_equal(expand.point(self.pd), (2, 2, 2))
        npt.assert_array_equal(irregular.vector(self.vd), (0.5, 2, 1))
        npt.assert_array_equal(irregular.point(self.pd), (0.5, 2, 1))

    def test_euler_rotation(self):
        id = EulerRotation([0, 0], [0, 1])

        npt.assert_array_equal(id.angle, 0.0)
        npt.assert_array_equal(id.rotation_axis, [np.nan, np.nan, np.nan])

        npt.assert_array_equal(id.point(self.px), self.px)
        npt.assert_array_equal(id.point(self.pd), self.pd)

        npt.assert_array_equal(id.matrix, np.eye(4))

    def test_translation(self):
        t = Translation((1, 2, 3))

        npt.assert_array_equal(t.displacement, [1, 2, 3])
        npt.assert_array_equal(t.matrix, np.asarray([[1, 0, 0, 0], [1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]]))
        npt.assert_array_equal((t @ t.inv).matrix, np.eye(4))

        npt.assert_array_equal(t * self.hpx, [1, 2, 2, 3])
        npt.assert_array_equal(t * self.hpd, [1, 2, 3, 4])
        npt.assert_array_equal(t * self.hvx, self.hvx)
        npt.assert_array_equal(t * self.hvd, self.hvd)
        npt.assert_array_equal(t.point(self.px), [2, 2, 3])
        npt.assert_array_equal(t.point(self.pd), [2, 3, 4])
        npt.assert_array_equal(t.vector(self.vx), self.vx)
        npt.assert_array_equal(t.vector(self.vd), self.vd)


if __name__ == '__main__':
    unittest.main()
