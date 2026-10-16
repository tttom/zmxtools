from typing import cast

import numpy as np
import pytest
from numpy import testing as npt

from zmxtools.optical_design.geometry import (
    IDENTITY,
    CompoundTransform,
    EulerRotation,
    Identity,
    InverseSphericalTransform,
    LiteralTransform,
    Positionable,
    Quaternion,
    Rotation,
    Scaling,
    SphericalTransform,
    Transform,
    Translation,
)


class TestGeometry:
    def setup_method(self):
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

        npt.assert_array_equal(
            identity.homogeneous([self.hpx, self.hvd, self.hpz]),
            [self.hpx, self.hvd, self.hpz],
        )
        npt.assert_array_equal(identity.vector([self.vx, self.vd]), [self.vx, self.vd])
        npt.assert_array_equal(identity.point([self.px, self.pd]), [self.px, self.pd])

        npt.assert_array_equal(
            identity.homogeneous([[self.hpx, self.hvd, self.hpz]]),
            np.asarray([[self.hpx, self.hvd, self.hpz]]),
        )
        npt.assert_array_equal(identity.vector([[self.vx, self.vd]]), np.asarray([[self.vx, self.vd]]))
        npt.assert_array_equal(identity.point([[self.px, self.pd]]), np.asarray([[self.px, self.pd]]))

        npt.assert_equal(identity @ identity == identity, True)

    def test_transform_base(self):
        t = Transform()
        with pytest.raises(NotImplementedError):
            _ = t.matrix
        with pytest.raises(NotImplementedError):
            t.homogeneous(self.hvx)
        with pytest.raises(NotImplementedError):
            _ = t.inv
        npt.assert_equal(str(t), "T")
        npt.assert_equal(repr(t), "Transform()")
        npt.assert_equal(hash(t), hash(repr(t)))
        npt.assert_equal(t == t, True)
        npt.assert_equal(t == 1, False)

        # Transforming by a scalar yields a Scaling compound; the base transform has no inverse
        with pytest.raises(NotImplementedError):
            t @ 2.0

        # A transform composed with its inverse is the identity
        translation = Translation((1, 2, 3))
        npt.assert_equal(translation @ translation.inv, IDENTITY)
        npt.assert_equal((translation.inv) @ translation, IDENTITY)

    def test_positionable(self):
        p = Positionable()
        with pytest.raises(NotImplementedError):
            p.to(IDENTITY)

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

    def test_scale_composition(self):
        shrink = Scaling(0.5)
        expand = Scaling(2)
        npt.assert_array_equal(cast("Scaling", shrink @ expand).scale, [1, 1, 1])
        npt.assert_array_equal(cast("Scaling", Scaling(2) @ Scaling(3)).scale, [6, 6, 6])
        npt.assert_array_equal((~Scaling(4)).scale, [0.25, 0.25, 0.25])
        npt.assert_equal(str(shrink).startswith("S"), True)
        npt.assert_equal(repr(shrink), "Scaling([0.5 0.5 0.5])")
        npt.assert_equal(shrink == Scaling(0.5), True)
        npt.assert_equal(shrink == expand, False)
        npt.assert_equal(shrink == "S", False)
        # Scaling composed with a non-scaling yields a compound
        npt.assert_equal(isinstance(Scaling(2) @ Translation((1, 0, 0)), CompoundTransform), True)

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

    def test_translation_composition(self):
        t1 = Translation((1, 2, 3))
        t2 = Translation((4, 5, 6))
        composed = t1 @ t2
        npt.assert_equal(isinstance(composed, Translation), True)
        npt.assert_array_equal(cast("Translation", composed).displacement, [5, 7, 9])
        npt.assert_array_equal((~t1).displacement, [-1, -2, -3])
        npt.assert_equal(str(t1).startswith("T"), True)
        npt.assert_equal(repr(t1), "Translation([1. 2. 3.])")
        npt.assert_equal(t1 == t2, False)
        npt.assert_equal(t1 == Translation((1, 2, 3)), True)
        npt.assert_equal(t1 == Scaling(1), False)
        # Translation composed with a non-translation yields a compound
        npt.assert_equal(isinstance(t1 @ Scaling(2), CompoundTransform), True)

    def test_literal_transform(self):
        m = np.eye(4)
        m[1:, 0] = [1, 2, 3]
        lt = LiteralTransform(m)
        npt.assert_array_equal(lt.matrix, m)
        npt.assert_array_equal(lt.point(self.origin), [1, 2, 3])
        npt.assert_array_equal(lt.vector(self.vd), self.vd)
        npt.assert_array_equal((~lt).matrix, np.linalg.inv(m))
        npt.assert_equal(str(lt).startswith("M("), True)
        npt.assert_equal(repr(lt).startswith("LiteralTransform("), True)
        npt.assert_equal(lt == LiteralTransform(m), True)
        npt.assert_equal(lt == Translation((1, 2, 3)), True)
        npt.assert_equal(lt == Scaling(1), False)

        # A 3x3 matrix is embedded in a 4x4 homogeneous matrix
        lt3 = LiteralTransform(np.eye(3))
        npt.assert_array_equal(lt3.matrix, np.eye(4))

    def test_compound_transform(self):
        t = Translation((1, 2, 3))
        s = Scaling(2)
        c = CompoundTransform(t, s)
        npt.assert_equal(len(c.components), 2)
        c.components = (s, t)
        npt.assert_array_equal(cast("Scaling", c.components[0]).scale, [2, 2, 2])
        npt.assert_array_equal(c.matrix, s.matrix @ t.matrix)
        npt.assert_array_equal(cast("Scaling", cast("CompoundTransform", ~c).components[1]).scale, [0.5, 0.5, 0.5])
        npt.assert_equal(str(c), str(s) + str(t))
        npt.assert_equal(repr(c).startswith("CompoundTransform("), True)
        npt.assert_equal(c == CompoundTransform(s, t), True)
        npt.assert_equal(c == t, False)
        npt.assert_equal(c == CompoundTransform(s, t, s), False)

        # Composition cancels inverse neighbors
        cancelled = CompoundTransform(t, t.inv) @ CompoundTransform(t, t.inv)
        npt.assert_equal(len(cancelled.components), 2)
        with_scalar = CompoundTransform(t, s) @ 3.0
        npt.assert_equal(isinstance(with_scalar, CompoundTransform), True)

    def test_euler_rotation(self):
        id = EulerRotation([0, 0], [0, 1])

        npt.assert_array_equal(id.angle, 0.0)
        npt.assert_array_equal(id.rotation_axis, [np.nan, np.nan, np.nan])

        npt.assert_array_equal(id.point(self.px), self.px)
        npt.assert_array_equal(id.point(self.pd), self.pd)

        npt.assert_array_equal(id.matrix, np.eye(4))

    def test_euler_rotation_angles_axes(self):
        e = EulerRotation([np.pi / 2, 0, 0], [0, 1, 2])
        npt.assert_array_equal(e.angles, [np.pi / 2, 0, 0])
        npt.assert_array_equal(e.axes, [0, 1, 2])
        inv = ~e
        npt.assert_array_equal(inv.angles, [-np.pi / 2, -0.0, -0.0])
        npt.assert_array_equal(inv.axes, [2, 1, 0])
        # Rotating by pi/2 around x maps y to z
        npt.assert_allclose(e.vector(self.vy), self.vz, atol=1e-12)
        npt.assert_allclose(e.point(self.py), self.pz, atol=1e-12)

    def test_quaternion(self):
        q = Quaternion()
        npt.assert_array_equal(q.scalar, 1)
        npt.assert_array_equal(q.vector, [0, 0, 0])
        npt.assert_array_equal(q.norm, 1)
        npt.assert_array_equal(q.norm2, 1)
        npt.assert_array_equal(q.vector_norm, 0)
        npt.assert_array_equal(q.vector_norm2, 0)
        npt.assert_array_equal(q[0], 1)
        npt.assert_array_equal(q[1:], [0, 0, 0])
        npt.assert_array_equal(q.angle, 0)
        npt.assert_array_equal(q.conj.values, [1, 0, 0, 0])
        npt.assert_array_equal((~q).values, [1, 0, 0, 0])
        npt.assert_array_equal(q.inv.values, [1, 0, 0, 0])
        npt.assert_array_equal(q.unit.values, [1, 0, 0, 0])
        npt.assert_equal(str(q), "1.0+0.0i+0.0j+0.0k")
        npt.assert_equal(repr(q).startswith("Quaternion("), True)
        npt.assert_equal(q == Quaternion(), True)
        npt.assert_equal(q == 1, False)

        # Arithmetic
        npt.assert_array_equal((q + q).values, [2, 0, 0, 0])
        npt.assert_array_equal((q - q).values, [0, 0, 0, 0])
        npt.assert_array_equal((-q).values, [-1, 0, 0, 0])
        npt.assert_array_equal((q * 2).values, [2, 0, 0, 0])
        npt.assert_array_equal((2 * q).values, [2, 0, 0, 0])
        npt.assert_array_equal((q / 2).values, [0.5, 0, 0, 0])
        npt.assert_array_equal((2 / q).values, [2, 0, 0, 0])
        npt.assert_array_equal((q * q).values, [1, 0, 0, 0])

        # exp and pow of the identity
        npt.assert_allclose(q.exp.values, [np.e, 0, 0, 0])
        npt.assert_allclose((q**2).values, [1, 0, 0, 0])

        # A 90-degree rotation around z
        q90 = Quaternion((np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))
        npt.assert_allclose(q90.angle, np.pi / 4)
        npt.assert_allclose((~q90).values, Quaternion((np.cos(np.pi / 4), 0, 0, -np.sin(np.pi / 4))).values)
        npt.assert_allclose(
            (q90 * q90).values, Quaternion((np.cos(np.pi / 2), 0, 0, np.sin(np.pi / 2))).values, atol=1e-12
        )
        npt.assert_allclose(q90.unit.values, q90.values)
        npt.assert_allclose((q90**2).values, (q90 * q90).values, atol=1e-12)
        npt.assert_allclose(
            q90.exp.values,
            np.exp(q90.scalar)
            * np.array([np.cos(q90.vector_norm), *(np.sin(q90.vector_norm) * q90.vector / q90.vector_norm)]),
        )

    def test_rotation(self):
        r = Rotation(rotation_axis=(0, 0, 1), angle=np.pi / 2)
        npt.assert_allclose(r.angle, np.pi / 2)
        npt.assert_allclose(r.rotation_axis, [0, 0, 1])
        npt.assert_allclose(r.vector(self.vx), self.vy, atol=1e-12)
        npt.assert_allclose(r.point(self.px), self.py, atol=1e-12)
        npt.assert_allclose(r.matrix @ np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1]), atol=1e-12)
        npt.assert_allclose((r.matrix @ np.array([0, 1, 0, 0]))[1:], self.vy, atol=1e-12)

        # The rotation axis length encodes the angle when no angle is given
        r2 = Rotation(rotation_axis=(0, 0, np.pi / 2))
        npt.assert_allclose(r2.angle, np.pi / 2, atol=1e-12)

        # Quaternion constructor
        r3 = Rotation(quaternion=(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))
        npt.assert_allclose(r3.angle, np.pi / 2, atol=1e-12)

        # Angle setter
        r4 = Rotation(rotation_axis=(0, 1, 0), angle=0.2)
        r4.angle = np.pi / 3
        npt.assert_allclose(r4.angle, np.pi / 3, atol=1e-12)

        # Rotation axis setter keeps the angle
        r5 = Rotation(rotation_axis=(0, 0, 1), angle=np.pi / 2)
        r5.rotation_axis = (1, 0, 0)
        npt.assert_allclose(r5.rotation_axis, [1, 0, 0], atol=1e-12)
        npt.assert_allclose(r5.angle, np.pi / 2, atol=1e-12)

        # Composition of rotations
        npt.assert_allclose(
            cast("Rotation", r @ Rotation(rotation_axis=(0, 0, 1), angle=np.pi / 2)).angle, np.pi, atol=1e-12
        )
        npt.assert_equal(r @ ~r, IDENTITY)
        npt.assert_equal(isinstance(r @ Scaling(2), CompoundTransform), True)
        npt.assert_equal(isinstance(r @ 2.0, CompoundTransform), True)
        npt.assert_allclose((~r).vector(self.vy), self.vx, atol=1e-12)
        npt.assert_equal(str(r).startswith("H"), True)
        npt.assert_equal(repr(r).startswith("Rotation("), True)
        npt.assert_equal(r == Rotation(rotation_axis=(0, 0, 1), angle=np.pi / 2), True)
        npt.assert_equal(r == Scaling(1), False)

        # Two opposite rotations compose to the identity
        npt.assert_equal(
            Rotation(rotation_axis=(0, 0, 1), angle=np.pi / 2) @ Rotation(rotation_axis=(0, 0, 1), angle=-np.pi / 2),
            IDENTITY,
        )

    def test_identity_matmul(self):
        npt.assert_equal(IDENTITY @ 2.0, Scaling(2))
        npt.assert_equal(2.0 @ IDENTITY, Scaling(2))
        npt.assert_equal(~IDENTITY, IDENTITY)
        npt.assert_equal(str(IDENTITY), "I")
        npt.assert_equal(repr(IDENTITY), "Identity()")
        npt.assert_equal(IDENTITY == Identity(), True)
        npt.assert_equal(IDENTITY == Scaling(1), True)
        npt.assert_equal(IDENTITY == Translation((1, 0, 0)), False)

    def test_spherical_transform(self):
        st = SphericalTransform(curvature=0.0)
        # Zero curvature is the planar limit
        npt.assert_allclose(st.point([0, 0, 1]), [np.pi, 0, 1], atol=1e-12)
        npt.assert_allclose(st.vector([0, 0, 1], coordinate=[0, 0, 0]), [0, 0, 1], atol=1e-12)
        npt.assert_equal(isinstance(~st, InverseSphericalTransform), True)

        st2 = SphericalTransform(curvature=0.5)
        p = st2.point([0, 0, 0])
        npt.assert_equal(p.shape[-1], 3)
        v = st2.vector([0, 0, 1], coordinate=[0, 0, 0])
        npt.assert_equal(v.shape[-1], 3)

    def test_inverse_spherical_transform(self):
        ist = InverseSphericalTransform(curvature=0.5)
        p = ist.point([0.1, 0.0, 0.0])
        npt.assert_equal(p.shape[-1], 3)
        v = ist.vector([0, 0, 1], coordinate=[0.1, 0.0, 0.0])
        npt.assert_equal(v.shape[-1], 3)
        npt.assert_equal(isinstance(~ist, SphericalTransform), True)

        # Round-trip through the spherical manifold for a point on the sphere
        st = SphericalTransform(curvature=0.5)
        pos = np.array([0.0, 0.0, 0.0])
        on_manifold = st.point(pos)
        back = ist.point(on_manifold)
        npt.assert_allclose(back, pos, atol=1e-9)
        pos2 = np.array([2 * np.sqrt(3) / 2, 0.0, 3.0])
        on_manifold2 = st.point(pos2)
        back2 = ist.point(on_manifold2)
        npt.assert_allclose(back2, pos2, atol=1e-9)
