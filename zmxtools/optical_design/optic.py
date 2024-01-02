from __future__ import  annotations

from typing import List, Sequence

from zmxtools import log

log = log.getChild(__name__)


class Surface:
    """A class to represent a thin surface between two volumes."""
    stop: bool = False
    description: str = ""  # A comment, often the name of the lens element
    curvature: float
    distance: float
    radius: float

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stop={self.stop}, description={self.description})"


class OpticalDesign:
    name: str = ""
    description: str = ""

    surfaces: List[Surface] = []
    wavelengths: Sequence[float] = []
    wavelength_weights: Sequence[float] = []
    material_libraries: List[MaterialLibrary] = []

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.name}, {self.description}, {repr(self.surfaces)}, "
                f"wavelengths={self.wavelengths}, wavelength_weights={self.wavelength_weights}, material_library={repr(self.material_libraries)})")


