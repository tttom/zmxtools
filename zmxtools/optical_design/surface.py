from __future__ import annotations

from typing import Optional, Self

import numpy as np

from zmxtools.optical_design import log
from zmxtools.optical_design.geometry import IDENTITY, Positionable, SphericalTransform, Transform
from zmxtools.optical_design.header import Element, Medium
from zmxtools.optical_design.light import LightPath, Wavefront
from zmxtools.utils.array import SCALAR_TYPE, array_like, array_type, asarray

log.getChild(__name__)


class Interface(Positionable):
    """A class to represent a surface interface interaction."""

    transform: Transform

    def __init__(self, transform: Transform = IDENTITY):
        """
        The position-specific transform of a plane normal to the z-axis to the curved interface surface.

        This transform should take vectors [0, 0, 1] to vectors that are normal to the interface and vectors that are
        orthogonal to [0, 0, 1] as orthogonal vectors tangent to the interface.
        """
        self.transform = transform

    def to(self, transform: Transform) -> Self:
        """Transform, or move, this interface by the specified transform."""
        self.transform = transform @ self.transform
        return self

    def refract_into(self, light_path: LightPath, medium: Medium) -> LightPath:
        """
        Refract from one medium to another at this interface.

        :param light_path: The light-path to refract at the interface should have its final Wavefront at the interface's
            surface.
        :param medium: The Medium at the back of the interface.

        :return: A lightpath that is one Wavefront longer than the input argument.
        """
        raise NotImplementedError

    def reflect(self, light_path: LightPath) -> LightPath:
        """
        Reflect from this interface.

        :param light_path: The light-path to refract at the interface should have its final Wavefront at the interface's
            surface.

        :return: A lightpath at is one Wavefront longer than the input argument.
        """
        light_path = light_path.to(self.transform.inv)  # This transform should make the z-axis normal to the interface.
        w = light_path.wavefront

        def flip(vector: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
            """Reverse the z-component. This assumes that the transform rotates the z-axis to the surface normal."""
            return asarray(vector) * asarray((1.0, 1.0, -1.0), float)

        return light_path.interact(electric_field=flip(w.electric_field), magnetizing_field=flip(w.magnetizing_field),
                                   k=flip(w.k), direction=flip(w.direction),
                                   ).to(self.transform)

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({repr(self.transform)})'


class SnellInterface(Interface):
    """Construct a new interface that refracts all light, i.e. without Fresnel reflections."""

    def refract_into(self, light_path: LightPath, medium: Medium) -> LightPath:
        """
        Refract from one medium to another at this interface using Snell's law.

        :param light_path: The light-path to refract at the interface should have its final Wavefront at the interface's
            surface.
        :param medium: The Medium at the back of the interface.

        :return: A lightpath at is one Wavefront longer than the input argument.
        """
        light_path = light_path.to(self.transform.inv)
        w = light_path.wavefront
        k_transverse = w.k[..., :2]
        k2_longitudinal = (w.k0 * medium.complex_refractive_index(
            wavenumber=w.k0, position=w.position,
            electric_field=w.electric_field, magnetizing_field=w.magnetizing_field,
        )) ** 2 - np.linalg.norm(k_transverse) ** 2
        transmitted = k2_longitudinal >= 0
        k2_longitudinal *= transmitted
        new_k = np.concatenate((k_transverse, k2_longitudinal ** 0.5), axis=-1)
        new_d = new_k / np.linalg.norm(new_k, axis=-1)  # TODO: update for non-isotropic materials
        return light_path.interact(k=new_k, direction=new_d, electric_field=w.electric_field,
                                   magnetizing_field=w.magnetizing_field,
                                   ).to(self.transform)


class Aperture(Positionable):
    """A class to represent an aperture that can selectively block the light."""

    transform: Transform

    def __init__(self, transform: Transform = IDENTITY):
        """
        The position-specific transform of a plane normal to the z-axis to the curved interface surface.

        This transform should take vectors [0, 0, 1] to vectors that are normal to the interface and vectors that are
        orthogonal to [0, 0, 1] as orthogonal vectors tangent to the interface.
        """
        self.transform = transform

    def to(self, transform: Transform) -> Self:
        """
        Moves or transforms this aperture.

        :param transform: The transform to apply to the left.

        :return: This aperture, but transformed.
        """
        self.transform = transform @ self.transform
        return self

    def transmit(self, light_path: LightPath) -> LightPath:
        """
        Computes a value to indicate whether points are within our outwith the aperture.

        :param light_path: The wavefront.

        :return: The signed distance from the aperture edge. Positive values indicate points inside, negative values
            indicate points outside the aperture.
        """
        return light_path

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({repr(self.transform)})'


INFINITE_APERTURE = Aperture()


class DiskAperture(Aperture):
    """A class to represent circular or annular apertures."""

    outer_radius: array_type[SCALAR_TYPE]
    inner_radius: array_type[SCALAR_TYPE]

    def __init__(self, outer_radius: array_like[SCALAR_TYPE] = np.inf, inner_radius: array_like[SCALAR_TYPE] = 0,
                 transform: Transform = IDENTITY,
                 ):
        """
        Construct a circular or annular aperture.

        :param outer_radius: The optional outer radius indicates outwith what radius light should be blocked.
        :param inner_radius: The optional inner radius indicates what light at the center should be blocked.
        :param transform: The transform to position and scale this aperture.
        """
        super().__init__(transform=transform)
        self.outer_radius = asarray(outer_radius)
        self.inner_radius = asarray(inner_radius)

    def transmit(self, light_path: LightPath) -> LightPath:
        """
        Computes a value to indicate whether points are within our outwith the aperture.

        :param light_path: The wavefront.

        :return: The signed distance from the aperture edge. Positive values indicate points inside, negative values
            indicate points outside the aperture.
        """
        light_path = light_path.to(self.transform.inv)
        w = light_path.wavefront
        r = np.linalg.norm(w[..., :2])
        inside = self.inner_radius <= r < self.outer_radius
        return light_path.interact(
            electric_field=w.electric_field * inside, magnetizing_field=w.magnetizing_field * inside,
        ).to(self.transform)

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({self.outer_radius}, {self.inner_radius}, {repr(self.transform)})'


class Surface(Element):
    """
    A base class to represent a thin surface between two volumes from which light can reflect, refract, or diffract.
    """

    interface: Interface
    aperture: Aperture

    def __init__(self, interface: Interface, aperture: Aperture = INFINITE_APERTURE, transform: Transform = IDENTITY):
        """
        Construct a new surface.

        :param interface: The material interface at this surface.
        :param aperture: The aperture of this surface.
        :param transform: The transform for the local coordinates of this surface.
        """
        super().__init__(transform=transform)
        self.interface = interface
        self.aperture = aperture

    def transmit_to(self, light_path: LightPath, medium: Medium) -> LightPath:
        """
        Propagate a light-field trough this surface to another medium.

        The final wavefront should already be at the front of this surface.

        :param light_path: The to-be-propagated light-field.
        :param medium: The medium to transmit to through this surface.

        :return: The light-field with the final wavefront at the other side of this surface.
        """
        light_path = light_path.to(self.transform.inv)
        light_path = self.aperture.transmit(light_path)
        light_path = self.interface.refract_into(light_path, medium)
        return light_path.to(self.transform)

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this object."""
        return f'{self.__class__.__name__}({repr(self.interface)}, {repr(self.aperture)}, {repr(self.transform)})'


class AnalyticSurface(Surface):
    """
    Marks a surface that can analytically calculate the distance.

    An analytic surface does not require iterative calls to the distance method.
    """


class PlanarSurface(AnalyticSurface):
    """A planar surface, normal to the z-axis in its local coordinate system as specified by self.transform."""

    def distance(self, wavefront: Wavefront) -> array_type[SCALAR_TYPE]:
        """The distance in units of wavefront.d to the intersection point of this element."""
        return - wavefront.position[..., 2] / wavefront.direction[..., 2]


class SphericalSurface(AnalyticSurface):
    """
    A simple spherical surface.

    TODO: Implement conic constant
    """

    def __init__(self, curvature: array_type[SCALAR_TYPE],
                 interface: Optional[Interface] = None, aperture: Optional[Aperture] = None,
                 transform: Transform = IDENTITY,
                 ):
        """
        Constructs a new spherical surface.

        :param curvature: The curvature, or 1 / radius of curvature, can be any real number. Positive means that the
             center of curvature is in the positive direction of the optical axis.
        :param interface: The interface at this surface.
        :param aperture: The aperture of this surface.
        :param transform: The transform from this surface to global coordinates.
        """
        self.curvature = curvature
        if aperture is None:
            aperture = DiskAperture(outer_radius=self.radius_of_curvature)
        interface.transform = SphericalTransform(self.curvature)
        super().__init__(interface=interface, aperture=aperture, transform=transform)

    @property
    def radius_of_curvature(self) -> array_type[SCALAR_TYPE]:
        """The radius of curvature of this surface."""
        return 1.0 / self.curvature

    def distance(self, wavefront: Wavefront) -> array_type[SCALAR_TYPE]:
        """Returns the analytic signed distance to the surface in units of the direction-vector."""
        wavefront = wavefront.to(self.transform.inv)
        p_rel_curv = wavefront.position * self.curvature - asarray([0, 0, 1])
        inp_p_d_curv = np.dot(p_rel_curv, wavefront.direction)
        norm2_d = np.linalg.norm(wavefront.direction) ** 2
        relative_signed_distance_sqd = inp_p_d_curv ** 2 - norm2_d * (np.linalg.norm(p_rel_curv) ** 2 - 1)
        distance_curv_in_units_of_d = (-inp_p_d_curv + np.sqrt(np.maximum(0, relative_signed_distance_sqd))) / norm2_d
        zero_curvature = self.curvature == 0
        return (distance_curv_in_units_of_d / (self.curvature + zero_curvature) +
                zero_curvature * (wavefront.position[2] / wavefront.direction[2])
                )
