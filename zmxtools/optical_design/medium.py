from __future__ import annotations

from dataclasses import dataclass

from zmxtools.optical_design import log
from zmxtools.optical_design.geometry import IDENTITY, Transform
from zmxtools.optical_design.header import Element, Medium
from zmxtools.optical_design.light import LightPath
from zmxtools.optical_design.material import Material
from zmxtools.utils.array import SCALAR_TYPEVAR, array_like, array_type

__all__ = ['log', 'HomogeneousMedium', 'InhomogeneousMedium']

log = log.getChild(__name__)


class HomogeneousMedium(Medium):
    """A class to represent a homogeneous medium."""

    def __init__(self, material: Material, transform: Transform = IDENTITY):
        """
        Construct a new medium from a material and a transform that locates the material in the medium.

        :param material: The material everywhere in the medium.
        :param transform: The origin and orientation of the material in the medium.
        """
        super().__init__(transform=transform)  # An anisotropic material may require a rotation.
        self.material: Material = material

    def propagate_to(self, light_path: LightPath, element: Element) -> LightPath:
        """Propagates on a line to the intersection. Only one new wavefront is added."""
        w = light_path.wavefront
        distance = element.distance(w)
        return light_path.propagate(distance)

    def complex_refractive_index(self, wavenumber: array_like[SCALAR_TYPEVAR],
                                 position: array_like[SCALAR_TYPEVAR],
                                 electric_field: array_like[SCALAR_TYPEVAR],
                                 magnetizing_field: array_like[SCALAR_TYPEVAR],
                                 ) -> array_type[SCALAR_TYPEVAR]:
        """The complex refractive index as a function of wavenumber, position, and the electromagnetic fields."""
        return self.material.complex_refractive_index(wavenumber=wavenumber)  # todo: implement birefringence

    def __repr__(self) -> str:
        """Returns a string that is a complete representation of this object."""
        return f'{self.__class__.__name__}({repr(self.material)}, {repr(self.transform)})'


@dataclass
class InhomogeneousMedium(Medium):
    """TODO: Implement GRIN media etc."""
