from __future__ import  annotations

from typing import Optional, Sequence, Self
import numpy as np

from zmxtools.optical_design.geometry import Transform, IDENTITY, Positionable
from zmxtools.utils.array import array_like, asarray, array_type, norm
from zmxtools.optical_design import log

log = log.getChild(__name__)


class Wavefront(Positionable):
    """
    A representation of a wavefront (or collection thereof) as a collection of rays.
    """
    def __init__(self,
                 E: array_like, H: array_like,
                 p: array_like = 0.0,
                 k: Optional[array_like] = None, d: Optional[array_like] = None,
                 k0: Optional[array_like] = None, ct: array_like = 0.0,
                 transform: Transform = IDENTITY):
        """
        Construct a wavefront by defining its rays as arrays of points, directions, and values.

        :param E: The electric field at each ray's position, relative to the coordinate_system if specified.
        :param H: The magnetizing field at each ray's position, relative to the coordinate_system if specified.
        :param p: The ray position, relative to the coordinate_system if specified.
        :param k: The wavevector of each ray at each ray's position, relative to the coordinate_system if specified.
        :param d: The direction of each ray, relative to the coordinate_system if specified.
        :param k0: The wavenumber in vacuum of each ray.
        :param ct: The equivalent optical path length along the ray in vacuum.
        :param transform: The transform to (lazily) apply to the input arguments.
        """
        self.__E = asarray(E, complex)
        self.__H = asarray(H, complex)
        self.__p = asarray(p, float)
        k = asarray(k, complex) if k is not None else asarray(k0, float)[..., np.newaxis] * asarray(d, complex)
        self.__k = k
        self.__d = asarray(d, float) if d is not None else k / norm(k)[..., np.newaxis]
        self.k0 = asarray(k0, float) if k0 is not None else norm(k)  # angular frequency in rad / m in vacuum
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
    def p(self) -> array_type:
        """The relative position of the definition of each ray."""
        return self.__transform.point(position=self.__p)

    @property
    def k(self) -> array_type:
        """The relative local wavevector: ||k|| / k0 = n"""
        return self.__transform.vector(position=self.__p, vector=self.__k)

    @property
    def d(self) -> array_type:
        """The relative ray direction (can be different from k for anisotropic materials)"""
        return self.__transform.vector(position=self.__p, vector=self.__d)

    @property
    def E(self) -> array_type:
        """
        Electric field density in the local coordinate system
        (can be non-orthogonal to k and H for anisotropic materials)
        """
        return self.__transform.vector(position=self.__p, vector=self.__E)

    @property
    def H(self) -> array_type:
        """
        The magnetizing field density in the local coordinate system
        (can be non-orthogonal to k and E for anisotropic materials)
        """
        return self.__transform.vector(vector=self.__H, position=self.__p)

    @property
    def shape(self) -> array_type:
        return asarray(np.broadcast_shapes(self.E.shape[:-1], self.H.shape[:-1],
                                           self.p.shape[:-1], self.k.shape[:-1], self.d.shape[:-1],
                                           self.k0.shape, self.ct.shape), int)

    @property
    def size(self) -> int:
        return np.prod(self.shape).item()

    @property
    def ndim(self) -> int:
        return self.shape.size

    @property
    def refractive_index(self) -> array_type:
        """The local real refractive index for each ray's wavelength, position, E, and H."""
        return norm(self.k) / self.k0


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
        super().__init__(E=self.wavefront.E, H=self.wavefront.H,
                         p=self.wavefront.p, k=self.wavefront.k, d=self.wavefront.d,
                         k0=self.wavefront.k0, ct=self.wavefront.ct)

    def to(self, transform: Transform) -> Self:
        """Transform all the properties of all the wavefronts."""
        self.__wavefronts = [_.to(transform) for _ in self.__wavefronts]
        return self

    @property
    def wavefronts(self) -> Sequence[Wavefront]:
        return tuple(self.__wavefronts)

    @property
    def wavefront(self) -> Wavefront:
        return self.wavefronts[-1]

    def append(self, wavefront: Wavefront) -> LightPath:
        self.__wavefronts.append(wavefront)
        return self

    def propagate(self, distance: array_like) -> LightPath:
        """
        Append a new wavefront that is propagated further by the specified distance (in units of  d ).

        :param distance: The distance in units of self.wavefront.d.

        :return: The updated LightPath, one wavefront longer than the current.
        """
        distance = asarray(distance, float)
        opd = distance * norm(self.wavefront.d) * self.wavefront.refractive_index
        return self.append(
            Wavefront(E=self.wavefront.E, H=self.wavefront.H,
                      p=self.wavefront.p + self.wavefront.d * distance,
                      k=self.wavefront.k, d=self.wavefront.d,
                      k0=self.wavefront.k0, ct=self.wavefront.ct + opd)
        )

    def interact(self,
                 E: Optional[array_like] = None, H: Optional[array_like] = None,
                 k: Optional[array_like] = None, d: Optional[array_like] = None) -> LightPath:
        """
        Append a new wavefront at the same position, p, but with different k, E, and H vectors.

        :param E: The new electric-field vector.
        :param H: The new magnetizing field vector.
        :param k: The new wavevector.
        :param d: The new direction vector.

        :return: The updated LightPath, one wavefront longer than the current.
        """
        w = self.wavefront
        if k is None:
            if d is None:
                d = w.d
            k = d / norm(d) * w.k0
        return self.append(
            Wavefront(E=w.E if E is None else E, H=w.H if H is None else H,
                      p=self.wavefront.p,
                      k=k, d=d,
                      k0=self.wavefront.k0, ct=self.wavefront.ct)
        )
