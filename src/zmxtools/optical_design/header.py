"""
A submodule to import top-level types from without causing cyclical import errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from zmxtools.optical_design import log
from zmxtools.optical_design.geometry import IDENTITY, Transform
from zmxtools.optical_design.light import LightPath, Wavefront
from zmxtools.utils.array import array_like, array_type

log = log.getChild(__name__)


class Element:
    """An abstract base class for all optical elements."""

    def __init__(self, transform: Transform = IDENTITY):
        """
        Construct a new optical element with a transform for its local coordinate system.

        Args:
            transform: The transform that takes global coordinates to local coordinates.

        """
        self.transform: Transform = transform

    def transmit_to(self, light: LightPath, medium: Medium) -> LightPath:
        """
        Propagates a light-path through this optical element into the specified medium behind it.

        Args:
            light: The light-path to extend.
            medium: The exit-medium in which the light-path is propagated after this optical element.

        Returns:
            A new light-path, with the final wavefront in the medium after this optical element.

        """
        raise NotImplementedError

    def reflect(self, light: LightPath) -> LightPath:
        """
        Returns the reflection of a light-path of of this optical element.

        Args:
            light: The incoming light-path before it hits this optical element.

        Returns:
            An extended light-path with the reflected wavefront at the end.

        """
        raise NotImplementedError

    def distance(self, wavefront: Wavefront) -> array_type:
        """
        The signed distance in units of wavefront.direction to the intersection point of this element.

        Negative values indicate that the wavefront is already past the surface.

        Args:
            wavefront: The wavefront of which the remaining travel distance is requested.

        Returns:
            An array with the distances between the wavefront and the first interface of this optical element.

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Returns a string that is a complete representation of this optical element."""
        return f"{self.__class__.__name__}({self.transform!r})"


@dataclass
class Medium:
    """A base class to represent a generic medium."""

    def __init__(self, transform: Transform = IDENTITY):
        """
        Base class for media that can consist of one or more materials.

        Materials can be anisotropic, so each medium has a coordinate system.

        Args:
            transform: The location of the coordinate system.

        """
        self.transform = transform

    def propagate_to(self, light_path: LightPath, element: Element) -> LightPath:
        """
        Propagates through this medium to the intersection with the specified optical element.

        Args:
            light_path: The light-path as it enters the medium.
            element: The element to propagate out to.

        Returns:
            The extended light-path, that reaches the front surface of the specified element.

        """
        raise NotImplementedError

    def complex_refractive_index(
        self,
        wavenumber: array_like,
        position: array_like,
        electric_field: array_like,
        magnetizing_field: array_like,
    ) -> array_type:
        """
        The complex refractive index as a function of wavenumber, position, and the electromagnetic fields.

        Its real part is the real refractive index, its imaginary part is the extinction coefficient. The latter is
        usually slightly negative for absorbing materials, though it can be negative for materials with gain.

        Arguments are broadcast as necessary.

        Args:
            wavenumber: The wavenumber, or numbers to compute the refractive index at.
            position: The position at which to compute the material properties.
            electric_field: The optional electric field, E, to compute the material properties for.
            magnetizing_field: The optional magnetizing field, H, to compute the material properties for.

        Returns:
            An array with the complex refractive index.

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Returns a string that is a complete representation of this medium."""
        return f"{self.__class__.__name__}({self.transform!r})"
