from __future__ import  annotations

from dataclasses import dataclass

from zmxtools.optical_design.header import Medium
from zmxtools.optical_design.material import Material
from zmxtools.optical_design.geometry import Transform, IDENTITY
from zmxtools.optical_design.light import LightPath
from zmxtools.utils.array import array_like, array_type
from zmxtools.optical_design import log

log = log.getChild(__name__)


class HomogeneousMedium(Medium):
    def __init__(self, material: Material, transform: Transform = IDENTITY):
        super().__init__(transform=transform)  # An anisotropic material may require a rotation.
        self.material: Material = material

    def propagate_to(self, light_path: LightPath, element) -> LightPath:
        """Propagates on a line to the intersection. Only one new wavefront is added."""
        w = light_path.wavefront
        distance = element.distance(w)
        return light_path.propagate(distance)

    def complex_refractive_index(self, wavenumber: array_like,
                                 p: array_like,
                                 E: array_like, H: array_like) -> array_type:
        return self.material.complex_refractive_index(wavenumber=wavenumber)  #, E=E, H=H)  # todo: implement birefringence

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.material)}, {repr(self.transform)})"


@dataclass
class InhomogeneousMedium(Medium):
    """
    TODO: Implement GRIN media etc.
    """
