from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

from zmxtools.optical_design import log
from zmxtools.optical_design.geometry import IDENTITY, Transform
from zmxtools.optical_design.header import Element, Medium
from zmxtools.optical_design.light import LightPath, Wavefront
from zmxtools.optical_design.source import Source
from zmxtools.optical_design.surface import Surface
from zmxtools.utils.array import SCALAR_TYPE, array_type

log = log.getChild(__name__)


class CompoundElement(Element):
    """A class to represent an optical element that consists of multiple sub-components."""

    def __init__(self, *elements_and_media: Element | Medium, transform: Transform = IDENTITY):
        """
        Construct a compound optical element from a series of elements, intercalated with media.

        :param elements_and_media: A sequence of N optical elements, with in-between N-1 media.
        :param transform: The transform of this element.
        """
        super().__init__(transform=transform)
        self.elements: Sequence[Element] = [_ for _ in elements_and_media if isinstance(_, Element)]
        self.media: Sequence[Medium] = [_ for _ in elements_and_media if isinstance(_, Medium)]
        self.media = self.media[:len(self.elements) - 1]

    def transmit_into(self, light: LightPath, medium: Medium) -> LightPath:
        """Propagates light as transmitted into the medium at the other side of this object."""
        for el, m, next_optic in zip(self.elements[:-1], self.media, self.elements[1:]):
            light = el.transmit_into(light, m)
            light = m.propagate_to(light, next_optic)
        return self.elements[-1].transmit_into(light, medium)  # trace out

    def distance(self, wavefront: Wavefront) -> array_type[SCALAR_TYPE]:
        """The distance in units of wavefront.d to the intersection point of this element."""
        return self.elements[0].distance(wavefront)

    def __repr__(self) -> str:
        """Returns a string that is a complete representation of this object."""
        optics_and_media = list(itertools.chain.from_iterable((o, m) for o, m in zip(self.elements, self.media)))
        optics_and_media.append(self.elements[-1])
        return f'{self.__class__.__name__}{tuple(optics_and_media)}'


@dataclass
class Detector:
    """A class to represent a detector that is immersed in a specific medium."""

    medium: Medium


@dataclass
class SurfaceDetector(Detector):
    """A class to represent a surface that acts as a detector."""

    surface: Surface


@dataclass
class OpticalDesign:
    """A class to represent a (compound) optical element as well as the test object (source) and image (detector)."""

    source: Source
    optic: Element
    detector: Detector
