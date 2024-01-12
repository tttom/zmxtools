"""
A submodule to import top-level types from without causing cyclical import errors.
"""
from __future__ import  annotations

from dataclasses import dataclass

from zmxtools.utils.array import array_like, array_type
from zmxtools.optical_design.geometry import Transform, IDENTITY
from zmxtools.optical_design.light import LightPath, Wavefront
from zmxtools.optical_design import log

log = log.getChild(__name__)


class Element:
    def __init__(self, transform: Transform = IDENTITY):
        self.transform: Transform = transform

    def transmit_into(self, light: LightPath, medium: Medium) -> LightPath:
        raise NotImplementedError

    def reflect(self, light: LightPath) -> LightPath:
        raise NotImplementedError

    def distance(self, wavefront: Wavefront) -> array_type:
        """
        The signed distance in units of wavefront.d to the intersection point of this element.
        Negative values indicate that the wavefront is already past the surface.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.transform)})"


@dataclass
class Medium:
    def __init__(self, transform: Transform = IDENTITY):
        """
        Base class for media that can consist of one or more materials.
        Materials can be anisotropic, so each medium has a coordinate system.

        :param transform: The location of the coordinate system.
        """
        self.transform = transform

    def propagate_to(self, light_path: LightPath, element) -> LightPath:
        raise NotImplementedError

    def complex_refractive_index(self, wavenumber: array_like,
                                 p: array_like,
                                 E: array_like, H: array_like) -> array_type:
        raise NotImplementedError
