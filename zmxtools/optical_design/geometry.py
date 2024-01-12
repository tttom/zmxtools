from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Self

from zmxtools.utils.array import array_like, asarray, array_type, stack, norm, sqrt, sin, cos, arctan2, dot, cross, einsum
from zmxtools.utils import script
from zmxtools.optical_design import log

log = log.getChild(__name__)


class Transform:
    def homogeneous(self, position: array_like, vector: array_like) -> array_type:
        """
        Apply this transform to a homogeneous vector or array of homogeneous vectors in the final (right-most) axis.
        The projective coordinate is element 0.

        :param position: The position in the homenous vector field.
        :param vector: The homogeneous 4-vector at each position.

        :return: The transformed homogeneous 4-vector at each position.
        """
        raise NotImplementedError

    def point(self, position: array_like) -> array_type:
        """
        Transform a 3D point, or array of points with the spatial dimension in the right-most axis.
        Rotations, Scalings, and Translations all affect points.

        :param position: The 3D-point (array) to be transformed.

        :return: The transformed 3-point at each position.
        """
        result = self.homogeneous(
            position=position,
            vector=np.concatenate((np.ones(shape=(position.shape[:-1], 1), dtype=position.dtype), position), axis=-1)
        )
        return result[..., 1:] / result[..., 0:1]

    def vector(self, position: array_like, vector: array_like) -> array_type:
        """
        Transform a 3D vector, or array of vectors with the spatial dimension in the right-most axis.
        Rotations and scalings affect vectors. Translations do not affect vectors, only points.

        :param position: The position in the homenous vector field.
        :param vector: The 3-vector at each position.

        :return: The transformed 3-vector at each position.
        """
        vector = asarray(vector)
        result = self.homogeneous(
            position=position,
            vector=np.concatenate((np.zeros(shape=(vector.shape[:-1], 1), dtype=vector.dtype), vector), axis=-1)
        )
        return result[..., 1:]

    def __invert__(self) -> Transform:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        raise NotImplementedError

    @property
    def inv(self) -> Transform:
        """
        Return the inverse of this transform. The inverse is also denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return ~self

    def __matmul__(self, right: Transform) -> Transform:
        """Combine multiple transformations into one. Simplifications are allowed."""
        if self.inv == right or self == right.inv:
            return IDENTITY
        return CompoundTransform(self, right)

    def __str__(self) -> str:
        return "T"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Transform) -> bool:
        return hash(self) == hash(other)


class Positionable:
    def to(self, transform: Transform) -> Self:
        """
        Transforms this object to a new position, or go from local to global coordinates.

        :param transform: The transform to the new position, or coordinate_system.inv.

        :return: The current object, transformed.
        """
        raise NotImplementedError
        return self


class HomogeneousTransform(Transform):
    @property
    def matrix(self) -> array_type:
        """
        The 4x4 homogeneous matrix corresponding to this transform.
        The homogeneous coordinate is the last (4th) dimension.
        """
        raise NotImplementedError

    def homogeneous(self, vector: array_like, position: array_like) -> array_type:
        """
        Apply this transform to a homogeneous vector or array of homogeneous vectors in the final (right-most) axis.
        The projective coordinate is element 0.
        """
        return self * vector

    def __mul__(self, homogeneous_vector: array_like):
        """
        Apply this transform to a homogeneous vector or array of homogeneous vectors in the final (right-most) axis.
        The projective coordinate is element 0.
        """
        return self.matrix @ asarray(homogeneous_vector)

    def __invert__(self) -> Transform:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return LiteralTransform(np.linalg.inv(self.matrix))

    def __eq__(self, other: Transform) -> bool:
        return (isinstance(other, HomogeneousTransform) and np.all(self.matrix == other.matrix)) or super() == other


class LiteralTransform(HomogeneousTransform):
    """Represents a generic homogeneous transform"""
    def __init__(self, matrix: array_like):
        """
        Construct a generic homogeneous transform using a 3D or 4D homogeneous matrix.

        :param matrix: The 4D homogeneous or 3D matrix. The projective coordinate is element 0.
        """
        matrix = asarray(matrix)
        if all(_ == 4 for _ in matrix.shape):
            self.__matrix = matrix
        else:
            self.__matrix = np.eye(4)
            self.__matrix[1:, 1:] = matrix

    @property
    def matrix(self) -> array_type:
        return self.__matrix

    def __str__(self) -> str:
        return f"M({self.matrix})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix})"

    def __eq__(self, other: Transform) -> bool:
        return (isinstance(other, LiteralTransform) and np.all(self.matrix == other.matrix)) or super() == other


class CompoundTransform(Transform):
    """A class to represent combinations of transforms."""
    def __init__(self, *components: Transform):
        """
        Construct a combination of transforms.

        :param components: The component transforms are executed in order from right-to-left.
        """
        self.__components = components

    @property
    def components(self) -> Sequence[Transform]:
        return self.__components

    @components.setter
    def components(self, new_components: Sequence[Transform]):
        self.__components = new_components

    @property
    def matrix(self) -> array_type:
        product = self.components[0].matrix
        for _ in self.components[1:]:
            product @= _.matrix
        return product

    def __matmul__(self, right: Transform) -> Transform:
        """Transforming a compound transform usually makes a larger composition."""
        self_components = self.components
        right_components = right.components if isinstance(right, CompoundTransform) else [right]
        if self_components[-1] == right_components[0].inv or self_components[-1].inv == right_components[0]:
            self_components = self_components[:-1]
            right_components = right_components[1:]
        return CompoundTransform(*self_components, *right_components)

    def __invert__(self) -> Transform:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return CompoundTransform(*(_.inv for _ in self.components[::-1]))

    def __str__(self) -> str:
        return "".join(str(_) for _ in self.components)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(repr(_) for _ in self.components)})"

    def __eq__(self, other: Transform) -> bool:
        return isinstance(other, CompoundTransform) and len(self.components) == len(other.components) and \
            all(s == o for s, o in zip(self.components, other.components))


class Translation(HomogeneousTransform):
    def __init__(self, displacement: array_like = (0.0, 0.0, 0.0)):
        """
        Construct a translation operation object.

        :param displacement: The displacement of points after translation.
        """
        self.__displacement = asarray((0.0, 0.0, 0.0), float)
        self.displacement = displacement

    @property
    def displacement(self) -> array_type:
        return self.__displacement

    @displacement.setter
    def displacement(self, new_displacement: array_like):
        self.__displacement = asarray(new_displacement, float)

    @property
    def matrix(self) -> array_type:
        m = np.eye(4)
        m[1:, 0] = self.displacement
        return m

    def __matmul__(self, right: Transform) -> Transform:
        """Translations applied to translations are still translations."""
        if isinstance(right, Translation):
            return Translation(displacement=self.displacement + right.displacement)
        return super().__matmul__(right)

    def __invert__(self) -> Translation:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return Translation(-self.displacement)

    def __str__(self) -> str:
        return f"T{script.super(','.join(str(_) for _ in self.displacement))}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.displacement})"

    def __eq__(self, other: Transform) -> bool:
        return (isinstance(other, Translation) and self.displacement == other.displacement) or super() == other


class Scaling(HomogeneousTransform):
    def __init__(self, scale: array_like = 1.0):
        """
        Construct a scaling operator.

        :param scale: A scalar scaling factor or a 3-vector, with a scale per dimension.
        """
        self.__scale = asarray((1.0, 1.0, 1.0), float)
        self.scale = scale

    @property
    def scale(self) -> array_type:
        return self.__scale

    @scale.setter
    def scale(self, new_scale: array_like):
        self.__scale[:] = new_scale

    @property
    def matrix(self) -> array_type:
        return np.diag((1.0, *self.scale))

    def __matmul__(self, right: Transform) -> Transform:
        """Scalings applied to Scalings are still Scalings."""
        if isinstance(right, Scaling):
            return Scaling(scale=self.scale + right.scale)
        return super().__matmul__(right)

    def __invert__(self) -> Scaling:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return Scaling(1.0 / self.scale)

    def __str__(self) -> str:
        return f"S{script.super(','.join(str(_) for _ in self.scale))}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.scale})"

    def __eq__(self, other: Transform) -> bool:
        return (isinstance(other, Scaling) and self.scale == other.scale) or super() == other


class Identity(Scaling):
    """The identity transform"""

    @property
    def matrix(self) -> array_type:
        return asarray(np.eye(4))

    def __matmul__(self, right: Transform) -> Transform:
        """The identity transform has no effect."""
        return right

    def __rmatmul__(self, left: Transform) -> Transform:
        """
        The identity transform has no effect.
        TODO: Is this ever called?
        """
        return left

    def __invert__(self) -> Identity:
        """The inverse of the identity is itself."""
        return self

    def __str__(self) -> str:
        return "I"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: Transform) -> bool:
        return isinstance(other, Identity) or super() == other


IDENTITY = Identity()


class Quaternion:
    """
    Represents a quaternion array.
    This class is used to implement RotationTransforms.
    """
    def __init__(self, values: array_like = (1.0, 0.0, 0.0, 0.0)):
        """Default: identity quaternion."""
        self.values = asarray(values, float)

    @property
    def scalar(self) -> array_type:
        return self.values[..., 0]

    @property
    def vector(self) -> array_type:
        return self.values[..., 1:]

    @property
    def norm(self) -> array_type:
        return np.linalg.norm(self.values, axis=-1)

    @property
    def norm2(self) -> array_type:
        return self.norm ** 2

    @property
    def vector_norm(self) -> array_type:
        return np.linalg.norm(self.vector, axis=-1)

    @property
    def vector_norm2(self) -> array_type:
        return self.vector_norm ** 2

    @property
    def unit(self) -> Quaternion:
        return self / self.norm

    @property
    def angle(self) -> array_type:
        """The angle of this quaternion."""
        return arctan2(self.vector_norm, self.scalar)

    @property
    def conj(self) -> Quaternion:
        return Quaternion(np.concatenate((self.scalar, -self.vector), axis=-1))

    def __getitem__(self, item) -> array_type:
        return self.values[item]

    def __add__(self, right: Quaternion) -> Quaternion:
        return Quaternion(self.values + right.values)

    def __neg__(self) -> Quaternion:
        return Quaternion(-self.values)

    def __sub__(self, right: Quaternion) -> Quaternion:
        return self + (-right)

    def __mul__(self, right: Quaternion | float) -> Quaternion:
        if isinstance(right, Quaternion):
            product = stack((
                self.scalar * right.scalar - np.dot(self.vector, right.vector),
                self.scalar * right[..., 1] + self[..., 1] * right.scalar + self[..., 2] * right[..., 3] - self[..., 3] * right[..., 2],
                self.scalar * right[..., 2] + self[..., 2] * right.scalar - self[..., 3] * right[..., 1] + self[..., 1] * right[..., 3],
                self.scalar * right[..., 3] + self[..., 3] * right.scalar + self[..., 1] * right[..., 2] - self[..., 2] * right[..., 1]
            ))
            return Quaternion(product)
        else:
            return Quaternion(self.values * right)

    def __rmul__(self, right: float) -> Quaternion:
        return self * right

    def __invert__(self) -> Quaternion:
        return Quaternion(self.conj.values / self.norm2)

    @property
    def inv(self) -> Quaternion:
        return ~self

    def __truediv__(self, right: Quaternion | float) -> Quaternion:
        return self * (1 / right)

    def __rdiv__(self, right: float) -> Quaternion:
        return right * ~self

    @property
    def exp(self) -> Quaternion:
        norm_vector = self.vector_norm
        values = np.exp(self.scalar) * asarray([np.cos(norm_vector), *(np.sin(norm_vector) * self.vector / norm_vector)])
        return Quaternion(values)

    def __pow__(self, power: float) -> Quaternion:
        return Quaternion(self.norm ** power * asarray([np.cos(self.angle * power),
                                                        *(np.sin(self.angle * power) * self.vector / self.vector_norm)]))

    def __str__(self) -> str:
        description = ''.join(f'{c:+}{v}' for c, v in zip(self.values, 'ijk'))
        if description.startswith("+"):
            description = description[1:]
        return description

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.values})"

    def __eq__(self, other: Quaternion) -> bool:
        return np.all(self.values == other.values)


class Rotation(HomogeneousTransform):
    def __init__(self, quaternion: Optional[Quaternion] | array_like = None,
                 rotation_axis: Optional[array_like] = None, angle: Optional[float] = None):
        """
        Create a new rotation object.

        :param quaternion: Four values, representing the scalar and vector part of a quaternion for half the rotation.
        :param rotation_axis: The optional rotation axis.
        :param angle: The optional rotation angle. If not specified, the length qof the axis is used as the angle in
            radians.
        """
        self.__quaternion = None
        if quaternion is None:
            rotation_axis = asarray(rotation_axis, float)
            if angle is None:
                angle = np.linalg.norm(rotation_axis)
                rotation_axis = rotation_axis / angle
            quaternion = np.cos(angle / 2.0), *(np.sin(angle / 2.0) * rotation_axis)
        self.quaternion = quaternion

    @property
    def quaternion(self) -> Quaternion:
        return self.__quaternion

    @quaternion.setter
    def quaternion(self, new_quaternion: Quaternion | array_like):
        if not isinstance(new_quaternion, new_quaternion):
            new_quaternion = Quaternion(new_quaternion).unit
        self.__quaternion = new_quaternion

    @property
    def angle(self) -> float:
        """The angle of rotation in radians, right-hand around the vector direction."""
        return self.quaternion.angle.item() * 2.0

    @angle.setter
    def angle(self, new_angle: float):
        self.quaternion = Quaternion(
            (np.cos(new_angle / 2.0),
             *(self.rotation_axis * np.sin(new_angle / 2.0))
             )
        )

    @property
    def rotation_axis(self) -> array_type:
        """The axis of rotation as a unit vector."""
        return self.quaternion.vector / self.quaternion.vector_norm

    @rotation_axis.setter
    def rotation_axis(self, new_axis: float):
        new_axis = asarray(new_axis, float)
        new_axis /= np.linalg.norm(new_axis)
        self.quaternion = (self.quaternion[0],
                           *(new_axis * self.quaternion.vector_norm)
                           )

    def __mul__(self, homogeneous_vector: array_like) -> array_type:
        homogeneous_vector = asarray(homogeneous_vector, float)
        homogeneous_vector[..., 0] = 0.0
        v = Quaternion(homogeneous_vector)
        product = self.quaternion * v / self.quaternion
        return product.values[1:]

    @property
    def matrix(self) -> array_type:
        return self * np.eye(4)

    def __matmul__(self, right: Transform) -> CompoundTransform | Rotation | Identity:
        """Combine multiple transformations into one. Simplifications are allowed."""
        if not isinstance(right, Rotation):
            return CompoundTransform(self, right)
        else:
            new_quaternion = self.quaternion * right.quaternion
            if new_quaternion.angle != 0.0:
                return Rotation(quaternion=new_quaternion)
            else:
                return Identity()

    def __invert__(self) -> Rotation:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return Rotation(quaternion=~self.quaternion)

    def __str__(self) -> str:
        return f"H{script.super(str(self.quaternion))}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.quaternion)})"

    def __eq__(self, other: Rotation) -> bool:
        return (isinstance(other, Rotation) and self.quaternion == other.quaternion) or super() == other


class EulerRotation(Rotation):
    def __init__(self, angles: array_like, axes: Sequence[int] = (0, 1, 2)):
        """
        Construct a rotation from a set of Euler angles in radians.

        :param angles: The consecutive rotation angles (in radians) around the Cartesian axes specified as  axes .
        :param axes: The order of the rotation axes, default (0, 1, 2): x, than y, than z.
        """
        self.__angles = asarray(angles, float)
        self.__axes = tuple(axes)

        half_angles = self.angles / 2.0
        q = Quaternion()
        for ha, axis in zip(half_angles, axes):
            q = q * Quaternion((np.cos(ha), *([0.0] * axis), np.sin(ha), *([0.0] * (2 - axis))))
        super().__init__(q)

    @property
    def angles(self) -> array_type:
        return self.__angles

    @property
    def axes(self) -> Sequence[int]:
        return self.__axes

    def __invert__(self) -> EulerRotation:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.
        This means that ~self @ self == identity == self @ ~self
        """
        return EulerRotation(angles=-self.angles, axes=self.axes[::-1])


class SphericalTransform(Transform):
    """
    Transforms from Cartesian coordinates to a spherical manifold that passes through the origin with its normal
    along the z-axis and with the given curvature.
    """
    def __init__(self, curvature: array_like):
        self.curvature = asarray(curvature, float)

    def point(self, position: array_like) -> array_type:
        """
        Transform a 3D point, or array of points with the spatial dimension in the right-most axis.
        Rotations, Scalings, and Translations all affect points.

        :param position: The to-be-transformed 3D-point at each position.

        :return: The transformed 3-point at each position.
        """
        position = asarray(position, float)

        transverse_radius = norm(position * asarray([1, 1, 0], float))
        radius_curv = sqrt((transverse_radius * self.curvature) ** 2 + (position[2] * self.curvature - 1.0) ** 2)  # always around self.curvature
        polar_distance_curv = arctan2(transverse_radius * self.curvature, position[2] * self.curvature - 1.0) * radius_curv  # always same sign as self.curvature
        azimuthal_angle = arctan2(position[1], position[0])  # in [-pi, pi)
        azimuthal_distance_curv = azimuthal_angle * polar_distance_curv
        delta_radius_curv = radius_curv - 1
        delta_radius_curv *= 2 * (position[2] * self.curvature - 1 >= 0) - 1  # positive is always in forward direction
        zero_curvature = self.curvature == 0.0  # Handle also planar interfaces

        return stack(polar_distance_curv, azimuthal_distance_curv, delta_radius_curv) / (self.curvature + zero_curvature) + \
            zero_curvature * stack(transverse_radius, azimuthal_angle * transverse_radius, position[..., 2])

    def vector(self, position: array_like, vector: array_like) -> array_type:
        """
        Transform a 3D vector, or array of vectors with the spatial dimension in the right-most axis.
        Rotations and scalings affect vectors. Translations do not affect vectors, only points.

        :param position: The position in the homenous vector field.
        :param vector: The to-be-transformed 3-vector at each position.

        :return: The transformed 3-vector at each position.
        """
        # build the coordinate systems on the manifold
        radial_curv = position * self.curvature - asarray([0, 0, 1], float)
        radial_curv *= 2 * (position[2] * self.curvature - 1 >= 0) - 1  # positive is always in forward direction
        radial = radial_curv / norm(radial_curv)
        transverse = position * asarray([1, 1, 0], float)
        transverse -= dot(transverse, radial) * radial
        zero_transverse = transverse == 0.0
        polar = transverse / (norm(transverse) + zero_transverse) + zero_transverse * asarray([1, 0, 0], float)
        azimuthal = cross(radial, polar)

        transformation_matrix = asarray([polar, azimuthal, radial], float)

        return einsum("i...j,...i->...j", transformation_matrix, vector)

    def __invert__(self) -> InverseSphericalTransform:
        return InverseSphericalTransform(curvature=self.curvature)


class InverseSphericalTransform(Transform):
    """
    Transforms from the spherical manifold back to Cartesian coordinates.
    """
    def __init__(self, curvature: array_type):
        self.curvature = asarray(curvature, float)

    def point(self, position: array_like) -> array_type:
        """
        Transform a 3D point, or array of points with the spatial dimension in the right-most axis.
        Rotations, Scalings, and Translations all affect points.

        :param position: The to-be-transformed 3D-point at each position.

        :return: The transformed 3-point at each position.
        """
        position = asarray(position, float)

        polar_distance = position[..., 0]
        azimuthal_distance = position[..., 1]
        relative_radius = position[..., 2]

        radius_curv = relative_radius * self.curvature + 1.0
        polar_angle = polar_distance * self.curvature / radius_curv
        azimuthal_angle = azimuthal_distance / polar_distance

        sin_polar_angle = sin(polar_angle)
        return (stack(cos(azimuthal_angle) * sin_polar_angle,
                      sin(azimuthal_angle) * sin_polar_angle,
                      cos(polar_angle)) * radius_curv - asarray([0, 0, 1], float)
                ) / self.curvature

    def vector(self, position: array_like, vector: array_like) -> array_type:
        """
        Transform a 3D vector, or array of vectors with the spatial dimension in the right-most axis.
        Rotations and scalings affect vectors. Translations do not affect vectors, only points.

        :param position: The position in the homenous vector field.
        :param vector: The to-be-transformed 3-vector at each position.

        :return: The transformed 3-vector at each position.
        """
        new_p = self.point(position)
        # build the coordinate systems on the manifold
        radial_curv = new_p * self.curvature - asarray([0, 0, 1])
        radial_curv *= 2 * (new_p[2] * self.curvature - 1 >= 0) - 1  # positive is always in forward direction
        radial = radial_curv / norm(radial_curv)
        transverse = new_p * asarray([1, 1, 0], float)
        transverse -= dot(transverse, radial) * radial
        zero_transverse = transverse == 0.0
        polar = transverse / (norm(transverse) + zero_transverse) + zero_transverse * asarray([1, 0, 0], float)
        azimuthal = cross(radial, polar)

        transformation_matrix = asarray([polar, azimuthal, radial])

        return einsum("j...i,...i->...j", transformation_matrix, vector)

    def __invert__(self) -> SphericalTransform:
        return SphericalTransform(curvature=self.curvature)


