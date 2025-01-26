from __future__ import annotations

from typing import Optional, Self, Sequence

import numpy as np

from zmxtools.optical_design import log
from zmxtools.utils import script
from zmxtools.utils.array import array_like, array_type, asarray

log = log.getChild(__name__)


# class HomogeneousCoordinates:
#     """A class to represent vectors in homogeneous coordinates."""
#
#     def __init__(self, vector: Optional[array_like] = None, point: Optional[array_like] = None, axis: int = 0):
#         self.axis = axis
#         if vector is not None:
#             if not isinstance(vector, np.ndarray):
#                 vector = asarray(vector)
#             vector = vector.swapaxes(0, self.axis)  # Internally always in the same axis.
#             if vector.shape[0] < 4:
#                 vector =
#         else:
#             if not isinstance(vector, np.ndarray):
#                 vector = asarray(vector)
#             vector = vector.swapaxes(0, self.axis)  # Internally always in the same axis.
#
#         if not isinstance(data, np.ndarray):
#             data = asarray(data)
#         self.data = data
#
#     def normalized(self) -> array_type:
#         return self.data / self.data[-1]
#
#
# array_like = array_like | HomogeneousCoordinates


class Transform:
    """A class to represent transforms."""

    def homogeneous(self, position: array_like, vector: array_like) -> array_type:
        """
        Apply this transform to a homogeneous vector or array of homogeneous vectors in the final (right-most) axis.

        The projective coordinate is element 0.

        :param position: The position in the homogeneous vector field.
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
            vector=np.concatenate((np.ones(shape=(*position.shape[:-1], 1), dtype=position.dtype), position), axis=-1),
        )
        return result[..., 1:] / result[..., 0:1]

    def vector(self, position: array_like, vector: array_like) -> array_type:
        """
        Transform a 3D vector, or array of vectors with the spatial dimension in the right-most axis.

        Rotations and scalings affect vectors. Translations do not affect vectors, only points.

        :param position: The position in the homogeneous vector field.
        :param vector: The 3-vector at each position.

        :return: The transformed 3-vector at each position.
        """
        vector = asarray(vector)
        result = self.homogeneous(
            position=position,
            vector=np.concatenate((np.zeros(shape=(*vector.shape[:-1], 1), dtype=vector.dtype), vector), axis=-1),
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
        """Return a string to display a transform."""
        return 'T'

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}()'

    def __hash__(self) -> int:
        """A relatively unique integer that can be used to check if two objects are not the same."""
        return hash(repr(self))

    def __eq__(self, other: Transform) -> bool:
        """Compares this transform with another, returning True when both transforms are numerically the same."""
        return repr(self) == repr(other)


class Positionable:
    """A mix-in to represent positionable objects that have a transform property."""

    def to(self, transform: Transform) -> Self:
        """
        Transforms this object to a new position, or go from local to global coordinates.

        :param transform: The transform to the new position, or coordinate_system.inv.

        :return: The current object, transformed.
        """
        raise NotImplementedError


class HomogeneousTransform(Transform):
    """A class to represent homogeneous transforms, i.e. those that can be represented by a 4x4 matrix."""

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
        """Compares this transform with another, returning True when both transforms are numerically the same."""
        return (isinstance(other, HomogeneousTransform) and np.all(self.matrix == other.matrix)) or super() == other


class LiteralTransform(HomogeneousTransform):
    """A class to represent generic homogeneous transforms."""

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
        """Returns the transformation matrix as an array."""
        return self.__matrix

    def __str__(self) -> str:
        """Return a string to display this matrix."""
        return f'M({self.matrix})'

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({self.matrix})'

    def __eq__(self, other: Transform) -> bool:
        """Compares this transform with another, returning True when both transforms are numerically the same."""
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
        """The individual component :py:class:``Transform`s as a Sequence."""
        return self.__components

    @components.setter
    def components(self, new_components: Sequence[Transform]):
        self.__components = new_components

    @property
    def matrix(self) -> array_type:
        """The matrix corresponding to this transform."""
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
        """Return a string to display this object."""
        return ''.join(str(_) for _ in self.components)

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f"{self.__class__.__name__}({', '.join(repr(_) for _ in self.components)})"

    def __eq__(self, other: Transform) -> bool:
        """Compares this transform with another, returning True when both transforms are numerically the same."""
        return (isinstance(other, CompoundTransform) and len(self.components) == len(other.components) and
                all(s == o for s, o in zip(self.components, other.components))
                )


class Translation(HomogeneousTransform):
    """A class to represent translations using homogeneous transforms."""

    def __init__(self, displacement: array_like = (0, 0, 0)):
        """
        Construct a translation operation object.

        :param displacement: The displacement of points after translation.
        """
        self.__displacement = asarray((0, 0, 0), float)
        self.displacement = displacement

    @property
    def displacement(self) -> array_type:
        """The translation vector incurred by this transformation."""
        return self.__displacement

    @displacement.setter
    def displacement(self, new_displacement: array_like):
        self.__displacement = asarray(new_displacement, float)

    @property
    def matrix(self) -> array_type:
        """The numerical representation of this transform as a 4x4 matrix."""
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
        """Return a string to display this object."""
        return f"T{script.sup(','.join(str(_) for _ in self.displacement))}"

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({self.displacement})'

    def __eq__(self, other: Transform) -> bool:
        """Compares this translation with another transform."""
        return (isinstance(other, Translation) and self.displacement == other.displacement) or super() == other


class Scaling(HomogeneousTransform):
    """A class to represent, isotropic or along the Cartesian axes."""

    def __init__(self, scale: array_like = 1.0):
        """
        Construct a scaling operator.

        :param scale: A scalar scaling factor or a 3-vector, with a scale per dimension.
        """
        self.__scale = asarray((1.0, 1.0, 1.0), float)
        self.scale = scale

    @property
    def scale(self) -> array_type:
        """The scaling in the 3 Cartesian dimensions incurred by this transformation."""
        return self.__scale

    @scale.setter
    def scale(self, new_scale: array_like):
        self.__scale[:] = new_scale

    @property
    def matrix(self) -> array_type:
        """The numerical representation of this transform as a 4x4 matrix."""
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
        """Return a string to display this object."""
        return f"S{script.sup(','.join(str(_) for _ in self.scale))}"

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({self.scale})'

    def __eq__(self, other: Transform) -> bool:
        """Compares this transform with another, returning True when both transforms are numerically the same."""
        return (isinstance(other, Scaling) and self.scale == other.scale) or super() == other


class Identity(Scaling):
    """A class to represente the identity transform."""

    @property
    def matrix(self) -> array_type:
        """The numerical representation of this transform as a 4x4 matrix."""
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
        """Return a string to represent the identity."""
        return 'I'

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}()'

    def __eq__(self, other: Transform) -> bool:
        """Compares this transform with another, returning True when both transforms are numerically the same."""
        return isinstance(other, Identity) or super() == other


IDENTITY = Identity()


class Quaternion:
    """
    Represents a quaternion array.

    This class is used to implement RotationTransforms.
    """

    def __init__(self, values: array_like = (1, 0, 0, 0)):
        """Default: identity quaternion."""
        self.values = asarray(values, float)

    @property
    def scalar(self) -> array_type:
        """The scalar component of this quaternion."""
        return self.values[..., 0]

    @property
    def vector(self) -> array_type:
        """The 3-element vector component of this quaternion."""
        return self.values[..., 1:]

    @property
    def norm(self) -> array_type:
        """The l2-norm of all values, scalar and vector."""
        return np.linalg.norm(self.values, axis=-1)

    @property
    def norm2(self) -> array_type:
        """The squared l2-norm of all values, scalar and vector."""
        return self.norm ** 2

    @property
    def vector_norm(self) -> array_type:
        """The l2-norm of the vector values only."""
        return np.linalg.norm(self.vector, axis=-1)

    @property
    def vector_norm2(self) -> array_type:
        """The squared l2-norm of the vector values only."""
        return self.vector_norm ** 2

    @property
    def unit(self) -> Quaternion:
        """Returns a normalized quaternion."""
        return self / self.norm

    @property
    def angle(self) -> array_type:
        """The angle of this quaternion."""
        return np.arctan2(self.vector_norm, self.scalar)

    @property
    def conj(self) -> Quaternion:
        """Returns the complex conjugate of this quaternion."""
        return Quaternion(np.concatenate((self.scalar, -self.vector), axis=-1))

    def __getitem__(self, item) -> array_type:
        """Returns the scalar components of the quaternion as indexed into an ndarray."""
        return self.values[item]

    def __add__(self, right: Quaternion) -> Quaternion:
        """Returns the sum of this quaternion and the specified one."""
        return Quaternion(self.values + right.values)

    def __neg__(self) -> Quaternion:
        """Returns the negative of this quaternion."""
        return Quaternion(-self.values)

    def __sub__(self, right: Quaternion) -> Quaternion:
        """Returns the difference of this quaternion and the specified one."""
        return self + (-right)

    def __mul__(self, right: Quaternion | float) -> Quaternion:
        """
        Multiplies this quaternion by another, or a simple scalar number.

        :param right: The quaternion on the right, or a scalar.

        :return: The product quaternion.
        """
        if isinstance(right, Quaternion):
            product = np.stack((
                self.scalar * right.scalar - np.dot(self.vector, right.vector),
                self.scalar * right[..., 1] +
                self[..., 1] * right.scalar + self[..., 2] * right[..., 3] - self[..., 3] * right[..., 2],
                self.scalar * right[..., 2] +
                self[..., 2] * right.scalar - self[..., 3] * right[..., 1] + self[..., 1] * right[..., 3],
                self.scalar * right[..., 3] +
                self[..., 3] * right.scalar + self[..., 1] * right[..., 2] - self[..., 2] * right[..., 1],
            ))
            return Quaternion(product)
        return Quaternion(self.values * right)

    def __rmul__(self, left: float) -> Quaternion:
        """Multiplies this quaternion by a scalar on the left, i.e. as on the right, returning a new quaternion."""
        return self * left

    def __invert__(self) -> Quaternion:
        """Returns the quaternion that is the inverse if this one."""
        return Quaternion(self.conj.values / self.norm2)

    @property
    def inv(self) -> Quaternion:
        """Returns the inverse of this quaternion."""
        return ~self

    def __truediv__(self, right: Quaternion | float) -> Quaternion:
        """
        Divides this quaternion by another, or a simple scalar number.

        :param right: The quaternion on the right, or a scalar.

        :return: The divided quaternion.
        """
        return self * (1 / right)

    def __rdiv__(self, left: float) -> Quaternion:
        """
        Returns the quaternion division of the value on the left by this quaternion.

        :param left: The value on the left.

        :return: The quaternion of which product with this equals the scalar value on the left.
        """
        return left * ~self

    @property
    def exp(self) -> Quaternion:
        """
        Compute the exponent of this quaternion, i.e. e-to-the-power-of-self.

        :return: exp(self)
        """
        norm_vector = self.vector_norm
        values = np.exp(self.scalar) * asarray(
            [np.cos(norm_vector), *(np.sin(norm_vector) * self.vector / norm_vector)],
        )
        return Quaternion(values)

    def __pow__(self, power: float) -> Quaternion:
        """
        Exponentiate this quaternion.

        :param power: The exponent, which can be fractional.

        :return: A new quaternion.
        """
        return Quaternion(self.norm ** power * asarray(
            [np.cos(self.angle * power),
             *(np.sin(self.angle * power) * self.vector / self.vector_norm),
             ],
        ))

    def __str__(self) -> str:
        """Return a string to display this quaternion."""
        description = ''.join(f'{c:+}{v}' for c, v in zip(self.values, 'ijk'))
        if description.startswith('+'):
            description = description[1:]
        return description

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({self.values})'

    def __eq__(self, other: Quaternion) -> bool:
        """Compares this quaternion with another one."""
        return np.all(self.values == other.values)


class Rotation(HomogeneousTransform):
    """A class to represent homogeneous rotatations."""

    def __init__(self, quaternion: Optional[Quaternion] | array_like = None,
                 rotation_axis: Optional[array_like] = None, angle: Optional[float] = None,
                 ):
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
        """Return the underlying quaternion."""
        return self.__quaternion

    @quaternion.setter
    def quaternion(self, new_quaternion: Quaternion | array_like):
        if not isinstance(new_quaternion, Quaternion):
            new_quaternion = Quaternion(new_quaternion)
        self.__quaternion = new_quaternion.unit

    @property
    def angle(self) -> float:
        """The angle of rotation in radians, right-hand around the vector direction."""
        return self.quaternion.angle.item() * 2.0

    @angle.setter
    def angle(self, new_angle: float):
        self.quaternion = Quaternion(
            (np.cos(new_angle / 2.0),
             *(self.rotation_axis * np.sin(new_angle / 2.0)),
             ),
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
                           *(new_axis * self.quaternion.vector_norm),
                           )

    def __mul__(self, homogeneous_vector: array_like) -> array_type:
        """
        Apply this transform to the specified vector and return a new, rotated vector.

        :param homogeneous_vector: The vector to rotate.

        :return: The rotated homogeneous vector.
        """
        homogeneous_vector = asarray(homogeneous_vector, float)
        homogeneous_vector[..., 0] = 0
        v = Quaternion(homogeneous_vector)
        product = self.quaternion * v / self.quaternion
        return product.values[1:]

    @property
    def matrix(self) -> array_type:
        """The numerical representation of this transform as a matrix."""
        return self * np.eye(4)

    def __matmul__(self, right: Transform) -> CompoundTransform | Rotation | Identity:
        """Combine multiple transformations into one. Simplifications are allowed."""
        if isinstance(right, Rotation):
            new_quaternion = self.quaternion * right.quaternion
            if new_quaternion.angle == 0:
                return IDENTITY
            return Rotation(quaternion=new_quaternion)
        return CompoundTransform(self, right)

    def __invert__(self) -> Rotation:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.

        This means that ~self @ self == identity == self @ ~self
        """
        return Rotation(quaternion=~self.quaternion)

    def __str__(self) -> str:
        """Return a string to display this object."""
        return f'H{script.sup(str(self.quaternion))}'

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({repr(self.quaternion)})'

    def __eq__(self, other: Rotation) -> bool:
        """Compares this rotation with another transform."""
        return (isinstance(other, Rotation) and self.quaternion == other.quaternion) or super() == other


class EulerRotation(Rotation):
    """A class to represent rotations around the Cartesian axes."""

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
            q = q * Quaternion((np.cos(ha), *(0 for _ in range(axis)), np.sin(ha), *(0 for _ in range(2 - axis))))
        super().__init__(q)

    @property
    def angles(self) -> array_type:
        """The angles by which to rotate around up to 3 axes, default: 0."""
        return self.__angles

    @property
    def axes(self) -> Sequence[int]:
        """The axes around which to rotate, default: 0, 1, 2."""
        return self.__axes

    def __invert__(self) -> EulerRotation:
        """
        Return the inverse of this transform. The inverse is denoted using the ~-operator.

        This means that ~self @ self == identity == self @ ~self
        """
        return EulerRotation(angles=-self.angles, axes=self.axes[::-1])


class SphericalTransform(Transform):
    """
    A class of objects that transform from Cartesian coordinates to a spherical manifold.

    The spherical manifold passes through the origin with its normal along the z-axis and with the given curvature.
    """

    def __init__(self, curvature: array_like):
        """
        Construct a transform from Cartesian coordinates to coordinates on a spherical manifold.

        :param curvature: The curvature of the manifold.
        """
        self.curvature = asarray(curvature, float)

    def point(self, position: array_like) -> array_type:
        """
        Transform a 3D point, or array of points with the spatial dimension in the right-most axis.

        Rotations, Scalings, and Translations all affect points.

        :param position: The to-be-transformed 3D-point at each position.

        :return: The transformed 3-point at each position.
        """
        position = asarray(position, float)

        transverse_radius = np.linalg.norm(position * asarray([1, 1, 0], float))
        radius_curv = np.sqrt((transverse_radius * self.curvature) ** 2 + (position[2] * self.curvature - 1.0) ** 2,
                              )  # always around self.curvature
        polar_distance_curv = np.arctan2(transverse_radius * self.curvature, position[2] * self.curvature - 1.0,
                                         ) * radius_curv  # always same sign as self.curvature
        azimuthal_angle = np.arctan2(position[1], position[0])  # in [-pi, pi)
        azimuthal_distance_curv = azimuthal_angle * polar_distance_curv
        delta_radius_curv = radius_curv - 1
        delta_radius_curv *= 2 * (position[2] * self.curvature - 1 >= 0) - 1  # positive is always in forward direction
        zero_curvature = self.curvature == 0  # Handle also planar interfaces

        return (np.stack(polar_distance_curv, azimuthal_distance_curv, delta_radius_curv,
                         ) / (self.curvature + zero_curvature) +
                zero_curvature * np.stack(transverse_radius, azimuthal_angle * transverse_radius, position[..., 2])
                )

    def vector(self, position: array_like, vector: array_like) -> array_type:
        """
        Transform a 3D vector, or array of vectors with the spatial dimension in the right-most axis.

        Rotations and scalings affect vectors. Translations do not affect vectors, only points.

        :param position: The position in the homogeneous vector field.
        :param vector: The to-be-transformed 3-vector at each position.

        :return: The transformed 3-vector at each position.
        """
        # build the coordinate systems on the manifold
        radial_curv = position * self.curvature - asarray([0, 0, 1], float)
        radial_curv *= 2 * (position[2] * self.curvature - 1 >= 0) - 1  # positive is always in forward direction
        radial = radial_curv / np.norm(radial_curv)
        transverse = position * asarray([1, 1, 0], float)
        transverse -= np.dot(transverse, radial) * radial
        zero_transverse = transverse == 0
        polar = transverse / (np.norm(transverse) + zero_transverse) + zero_transverse * asarray([1, 0, 0], float)
        azimuthal = np.cross(radial, polar)

        transformation_matrix = asarray([polar, azimuthal, radial], float)

        return np.einsum('i...j,...i->...j', transformation_matrix, vector)

    def __invert__(self) -> InverseSphericalTransform:
        """Returns the transform to which this one is the inverse, i.e. the inverse of this one."""
        return InverseSphericalTransform(curvature=self.curvature)


class InverseSphericalTransform(Transform):
    """A class to represent a transform from the spherical manifold back to Cartesian coordinates."""

    def __init__(self, curvature: array_type):
        """
        Construct a spatially variant transform to go from coordinates on a spherical surface to Cartesian coordinates.

        :param curvature: The curvature of the manifold.
        """
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

        sin_polar_angle = np.sin(polar_angle)
        return (np.stack(np.cos(azimuthal_angle) * sin_polar_angle,
                         np.sin(azimuthal_angle) * sin_polar_angle,
                         np.cos(polar_angle),
                         ) * radius_curv - asarray([0, 0, 1], float)
                ) / self.curvature

    def vector(self, position: array_like, vector: array_like) -> array_type:
        """
        Transform a 3D vector, or array of vectors with the spatial dimension in the right-most axis.

        Rotations and scalings affect vectors. Translations do not affect vectors, only points.

        :param position: The position in the homogeneous vector field.
        :param vector: The to-be-transformed 3-vector at each position.

        :return: The transformed 3-vector at each position.
        """
        new_p = self.point(position)
        # build the coordinate systems on the manifold
        radial_curv = new_p * self.curvature - asarray([0, 0, 1])
        radial_curv *= 2 * (new_p[2] * self.curvature - 1 >= 0) - 1  # positive is always in forward direction
        radial = radial_curv / np.linalg.norm(radial_curv)
        transverse = new_p * asarray([1, 1, 0], float)
        transverse -= np.dot(transverse, radial) * radial
        zero_transverse = transverse == 0
        polar = transverse / (np.linalg.norm(transverse) + zero_transverse) + zero_transverse * asarray([1, 0, 0])
        azimuthal = np.cross(radial, polar)

        transformation_matrix = asarray([polar, azimuthal, radial])

        return np.einsum('j...i,...i->...j', transformation_matrix, vector)

    def __invert__(self) -> SphericalTransform:
        """Returns a new transform that is the inverse if this one."""
        return SphericalTransform(curvature=self.curvature)
