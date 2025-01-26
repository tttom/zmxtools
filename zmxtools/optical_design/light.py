from __future__ import annotations

from typing import Optional, Self, Sequence

import numpy as np

from zmxtools.optical_design import log
from zmxtools.optical_design.geometry import IDENTITY, Positionable, Transform
from zmxtools.utils.array import array_like, array_type, asarray

log = log.getChild(__name__)


class Wavefront(Positionable):
    """A representation of a wavefront (or collection thereof) as a collection of rays."""

    def __init__(self,
                 electric_field: array_like, magnetizing_field: array_like,
                 position: array_like = 0,
                 k: Optional[array_like] = None, direction: Optional[array_like] = None,
                 k0: Optional[array_like] = None, ct: array_like = 0,
                 transform: Transform = IDENTITY,
                 ):
        """
        Construct a wavefront by defining its rays as arrays of points, directions, and values.

        :param electric_field: The electric field at each ray's position, relative to the specified coordinate_system.
        :param magnetizing_field: The magnetizing field at each ray's position, relative to the coordinate_system.
        :param position: The ray position, relative to the coordinate_system if specified.
        :param k: The wavevector of each ray at each ray's position, relative to the coordinate_system if specified.
        :param direction: The direction of each ray, relative to the coordinate_system if specified.
        :param k0: The wavenumber in vacuum of each ray.
        :param ct: The equivalent optical path length along the ray in vacuum.
        :param transform: The transform to (lazily) apply to the input arguments.
        """
        self.__E = asarray(electric_field, complex)
        self.__H = asarray(magnetizing_field, complex)
        self.__p = asarray(position, float)
        k = asarray(k, complex) if k is not None else asarray(k0, float)[..., np.newaxis] * asarray(direction, complex)
        self.__k = k
        self.__d = asarray(direction, float) if direction is not None else k / np.linalg.norm(k)[..., np.newaxis]
        self.k0 = asarray(k0, float) if k0 is not None else np.linalg.norm(k)  # angular frequency in rad / m in vacuum
        self.ct = asarray(ct, float)  # Optical path difference in vacuum in meters

        self.__transform: Transform = transform

    def to(self, transform: Transform) -> Self:
        """
        Get a new wavefront representation, but relative to the specified coordinate system.

        :param transform: The transform to apply to all points and vectors.

        :return: The wavefront with the transformed properties.
        """
        self.__transform = transform @ self.__transform
        return self

    @property
    def position(self) -> array_type:
        """The relative position of the definition of each ray."""
        return self.__transform.point(position=self.__p, coordinate=self.__p)

    @property
    def k(self) -> array_type:
        """The relative local wavevector: ||k|| / k0 = n."""
        return self.__transform.vector(vector=self.__k, coordinate=self.__p)

    @property
    def direction(self) -> array_type:
        """The relative ray direction, which can be different from k for anisotropic materials."""
        return self.__transform.vector(vector=self.__d, coordinate=self.__p)

    @property
    def electric_field(self) -> array_type:
        """
        Electric field density, E, in the local coordinate system.

        This coordinate system can be non-orthogonal to k and H for anisotropic materials)
        """
        return self.__transform.vector(vector=self.__E, coordinate=self.__p)

    @property
    def magnetizing_field(self) -> array_type:
        """
        The magnetizing field, H, density in the local coordinate system.

        This coordinate system can be non-orthogonal to k and E for anisotropic materials.
        """
        return self.__transform.vector(vector=self.__H, coordinate=self.__p)

    @property
    def shape(self) -> array_type:
        """The shape of the ray bundle that makes up the wavefront."""
        return asarray(np.broadcast_shapes(self.electric_field.shape[:-1], self.magnetizing_field.shape[:-1],
                                           self.position.shape[:-1], self.k.shape[:-1], self.direction.shape[:-1],
                                           self.k0.shape, self.ct.shape,
                                           ),
                       int,
                       )

    @property
    def size(self) -> int:
        """The number of rays in the wavefront."""
        return np.prod(self.shape).item()

    @property
    def ndim(self) -> int:
        """The number of dimensions that the wavefront rays are packed in."""
        return self.shape.size

    @property
    def refractive_index(self) -> array_type:
        """The local real refractive index for each ray's wavelength, position, E, and H."""
        return np.linalg.norm(self.k) / self.k0


class LightPath(Positionable):
    """
    A representation of a lightpath (or collection thereof) as a collection of rays and their history.

    A LightPath is a drop-in replacement for its last wavefront.
    """

    def __init__(self, *wavefronts: Wavefront):
        """
        Construct a wavefront by defining its rays as arrays of points, directions, and values.

        :param wavefronts: The wavefronts of this light path.
        """
        self.__wavefronts = list(wavefronts)  # All the wavefronts from the start of the rays to the end.
        self.__transform: Transform = IDENTITY
        super().__init__(electric_field=self.wavefront.electric_field,
                         magnetizing_field=self.wavefront.magnetizing_field,
                         position=self.wavefront.position, k=self.wavefront.k, direction=self.wavefront.direction,
                         k0=self.wavefront.k0, ct=self.wavefront.ct,
                         )

    def to(self, transform: Transform) -> Self:
        """Transform all the properties of all the wavefronts."""
        self.__wavefronts = [_.to(transform) for _ in self.__wavefronts]
        return self

    @property
    def wavefronts(self) -> Sequence[Wavefront]:
        """The wavefronts that make up this light-path."""
        return tuple(self.__wavefronts)

    @property
    def wavefront(self) -> Wavefront:
        """The, currently, final wavefront of this light-path."""
        return self.wavefronts[-1]

    def append(self, wavefront: Wavefront) -> LightPath:
        """Extend this light-path by appending one wavefront to the end."""
        self.__wavefronts.append(wavefront)
        return self

    def propagate(self, distance: array_like) -> LightPath:
        """
        Append a new wavefront that is propagated further by the specified distance (in units of  d ).

        :param distance: The distance in units of self.wavefront.d.

        :return: The updated LightPath, one wavefront longer than the current.
        """
        distance = asarray(distance, float)
        opd = distance * np.linalg.norm(self.wavefront.direction) * self.wavefront.refractive_index
        return self.append(
            Wavefront(electric_field=self.wavefront.electric_field, magnetizing_field=self.wavefront.magnetizing_field,
                      position=self.wavefront.position + self.wavefront.direction * distance,
                      k=self.wavefront.k, direction=self.wavefront.direction,
                      k0=self.wavefront.k0, ct=self.wavefront.ct + opd,
                      ),
        )

    def interact(self,
                 electric_field: Optional[array_like] = None, magnetizing_field: Optional[array_like] = None,
                 k: Optional[array_like] = None, direction: Optional[array_like] = None,
                 ) -> LightPath:
        """
        Append a new wavefront at the same position, p, but with different k, E, and H vectors.

        :param electric_field: The new electric-field vector.
        :param magnetizing_field: The new magnetizing field vector.
        :param k: The new wavevector.
        :param direction: The new direction vector.

        :return: The updated LightPath, one wavefront longer than the current.
        """
        w = self.wavefront
        if k is None:
            if direction is None:
                direction = w.direction
            k = direction / np.linalg.norm(direction) * w.k0
        return self.append(
            Wavefront(electric_field=w.electric_field if electric_field is None else electric_field,
                      magnetizing_field=w.magnetizing_field if magnetizing_field is None else magnetizing_field,
                      position=self.wavefront.position,
                      k=k, direction=direction,
                      k0=self.wavefront.k0, ct=self.wavefront.ct,
                      ),
        )
