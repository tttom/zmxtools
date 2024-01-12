from __future__ import annotations

import numpy as np
from typing import Optional, Self

from zmxtools.optical_design.medium import Medium
from zmxtools.utils.array import array_like, asarray, array_type, norm, norm2, dot, sqrt, maximum
from zmxtools.optical_design.geometry import Transform, IDENTITY, Positionable, SphericalTransform
from zmxtools.optical_design.optic import Element
from zmxtools.optical_design.light import LightPath, Wavefront
from zmxtools.optical_design import log

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

        def flip(vector: array_type) -> array_type:
            """Reverse the z-component. This assumes that the transform rotates the z-axis to the surface normal."""
            return vector * asarray([1.0, 1.0, -1.0], float)

        return light_path.interact(E=flip(w.E), H=flip(w.H), k=flip(w.k), d=flip(w.d)).to(self.transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.transform)})"


class SnellInterface(Interface):
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
        k2_longitudinal = (w.k0 * medium.complex_refractive_index(wavenumber=w.k0, p=w.p, E=w.E, H=w.H)) ** 2 - norm(k_transverse) ** 2
        transmitted = k2_longitudinal >= 0.0
        k2_longitudinal *= transmitted
        new_k = np.concatenate((k_transverse, k2_longitudinal ** 0.5), axis=-1)
        new_d = new_k / np.linalg.norm(new_k, axis=-1)  # TODO: update for non-isotropic materials
        return light_path.interact(k=new_k, d=new_d, E=w.E, H=w.H).to(self.transform)


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
        self.transform = transform @ self.transform
        return self

    def transmit(self, light_path: LightPath) -> LightPath:
        """
        Compute the distance from the center of the aperture. A value of

        :param light_path: The wavefront.

        :return: The signed distance from the aperture edge. Positive values indicate points inside, negative values
            indicate points outside the aperture.
        """
        return light_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.transform)})"


INFINITE_APERTURE = Aperture()


class DiskAperture(Aperture):
    outer_radius: array_type
    inner_radius: array_type

    def __init__(self, outer_radius: array_like = np.inf, inner_radius: array_like = 0.0,
                 transform: Transform = IDENTITY):
        super().__init__(transform=transform)
        self.outer_radius = asarray(outer_radius)
        self.inner_radius = asarray(inner_radius)

    def transmit(self, light_path: LightPath) -> LightPath:
        light_path = light_path.to(self.transform.inv)
        w = light_path.wavefront
        r = norm(w[..., :2])
        inside = self.inner_radius <= r < self.outer_radius
        return light_path.interact(E=w.E * inside, H=w.H * inside).to(self.transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.outer_radius}, {self.inner_radius}, {repr(self.transform)})"


class Surface(Element):
    """A class to represent a thin surface between two volumes from which light can reflect, refract, or diffract."""
    interface: Interface
    aperture: Aperture

    def __init__(self, interface: Interface, aperture: Aperture = INFINITE_APERTURE, transform: Transform = IDENTITY):
        super().__init__(transform=transform)
        self.interface = interface
        self.aperture = aperture

    def transmit_to(self, light_path: LightPath, medium: Medium) -> LightPath:
        light_path = light_path.to(self.transform.inv)
        light_path = self.aperture.transmit(light_path)
        light_path = self.interface.refract_into(light_path, medium)
        return light_path.to(self.transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.interface)}, {repr(self.aperture)}, {repr(self.transform)})"


class AnalyticSurface(Surface):
    """
    Marks a surface that can analytically calculate the distance. As such, no iterative call's to the distance
     method are required.
    """


class PlanarSurface(AnalyticSurface):
    """A planar surface, normal to the z-axis in its local coordinate system as specified by self.transform."""
    def distance(self, wavefront: Wavefront) -> array_type:
        """The distance in units of wavefront.d to the intersection point of this element."""
        return - wavefront.p[..., 2] / wavefront.d[..., 2]


class SphericalSurface(AnalyticSurface):
    """
    A simple spherical surface.
    TODO: Implement conic constant
    """
    def __init__(self, curvature: array_type,
                 interface: Optional[Interface] = None, aperture: Optional[Aperture] = None,
                 transform: Transform = IDENTITY):
        self.curvature = curvature
        if aperture is None:
            aperture = DiskAperture(outer_radius=self.radius_of_curvature)
        interface.transform = SphericalTransform(self.curvature)
        super().__init__(interface=interface, aperture=aperture, transform=transform)

    @property
    def radius_of_curvature(self) -> array_type:
        return 1.0 / self.curvature

    def distance(self, wavefront: Wavefront) -> array_type:
        """Returns the analytic signed distance to the surface in units of the d-vector."""
        wavefront = wavefront.to(self.transform.inv)
        p_rel_curv = wavefront.p * self.curvature - asarray([0, 0, 1])
        inp_p_d_curv = dot(p_rel_curv, wavefront.d)
        norm2_d = norm2(wavefront.d)
        # norm2(p_rel + wavefront.d * x) * curvature ** 2 == 1
        # norm2_d * x_curv**2 + 2 * inp_p_d_curv * x_curv + norm2(p_rel_curv) - 1  == 0
        distance_curv_in_units_of_d = (-inp_p_d_curv + sqrt(maximum(0.0, inp_p_d_curv ** 2 - norm2_d * (norm2(p_rel_curv) - 1.0)))) / norm2_d
        zero_curvature = self.curvature == 0.0
        return distance_curv_in_units_of_d / (self.curvature + zero_curvature) + \
            zero_curvature * (wavefront.p[2] / wavefront.d[2])
