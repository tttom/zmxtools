from __future__ import  annotations

from dataclasses import dataclass, field
from typing import List, Sequence
import itertools

from zmxtools.optical_design.geometry import Transform, IDENTITY
from zmxtools.optical_design.header import Medium, Element
from zmxtools.optical_design.surface import Surface
from zmxtools.optical_design.light import Wavefront, LightPath
from zmxtools.optical_design.source import Source
from zmxtools.utils.array import array_type
from zmxtools.optical_design import log

log = log.getChild(__name__)


class CompoundElement(Element):
    def __init__(self, *elements_and_media: Element | Medium, transform: Transform = IDENTITY):
        super().__init__(transform=transform)
        self.elements: Sequence[Element] = [_ for _ in elements_and_media if isinstance(_, Element)]
        self.media: Sequence[Medium] = [_ for _ in elements_and_media if isinstance(_, Medium)]
        self.media = self.media[:len(self.elements) - 1]

    def transmit_into(self, light, medium: Medium) -> LightPath:
        for element, medium, next_optic in zip(self.elements[:-1], self.media, self.elements[1:]):
            light = element.transmit_into(light, medium)
            light = medium.propagate_to(light, next_optic)
        light = self.elements[-1].transmit_into(light, medium)  # trace out
        return light

    def distance(self, wavefront: Wavefront) -> array_type:
        """The distance in units of wavefront.d to the intersection point of this element."""
        return self.elements[0].distance(wavefront)

    def __repr__(self) -> str:
        optics_and_media = list(itertools.chain.from_iterable((o, m) for o, m in zip(self.elements, self.media)))
        optics_and_media.append(self.elements[-1])
        for _ in optics_and_media:
            print(repr(_))
        return f"{self.__class__.__name__}{tuple(optics_and_media)}"


@dataclass
class Detector:
    medium: Medium


@dataclass
class SurfaceDetector(Detector):
    surface: Surface


@dataclass
class OpticalDesign:
    """
    A class to represent a (compound) optical element as well as the test object (source) and image (detector).
    """
    source: Source
    optic: Element
    detector: Detector


