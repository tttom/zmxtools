from __future__ import  annotations

import re
import typing
from typing import Optional, Self, List, Sequence, Union, Iterator
import pathlib
import io
import os
from dataclasses import dataclass
import numpy as np

from zmxtools import log

log = log.getChild(__name__)

array_type = np.ndarray
array_like = array_type | int | float | complex | Sequence["array_like"]
asarray = lambda _: np.asarray(_, dtype=np.complex64)

const_c = 299_792_458  # avoid importing scipy just for this


class BytesFile:
    """A class to represent bytes as a file stream without it coming from disk."""
    def __init__(self, path: pathlib.Path | str, contents: Optional[bytes] = None):
        if isinstance(path, str):
            path = pathlib.Path(path)
        if contents is None:
            with open(path, "rb") as f:
                contents = f.read()
        self.__content_bytes: bytes = contents
        self.__content_stream: Optional[io.BytesIO] = None
        self.__path: pathlib.Path = path

    @property
    def path(self) -> pathlib.Path:
        """The path of this file can be ficticious."""
        return self.__path

    @property
    def name(self) -> str:
        """The name of this file."""
        return self.path.as_posix()

    def open(self) -> Self:
        self.__content_stream = io.BytesIO(self.__content_bytes)
        return self

    def close(self):
        self.__content_stream = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # Do not suppress Exceptions

    def read(self, n: int = -1) -> bytes:
        """Read like a typing.BinaryIO object."""
        if self.__content_stream is None:
            with self:
                return self.read(n)
        else:
            return self.__content_stream.read(n)


BinaryFileLike = Union[BytesFile, typing.BinaryIO]
FileLike = Union[BinaryFileLike, typing.IO]
PathLike = Union[os.PathLike, str]


@dataclass
class MaterialResistanceValue:
    value: float = 2
    limits: Sequence[float] = (0, 5)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __str__(self) -> str:
        descriptions = ["high", "medium-high", "medium", "medium-low", "low"]
        very_low_resistances = ["very-low", "very-low", "very-very-low", "extremely-low", "too-low"]
        if min(self.limits) <= self.value <= 25:
            if self.limits[0] > 0:
                description = descriptions[max(0, min(int(self.value) - 1, len(descriptions) - 1))]
            else:
                description = descriptions[max(0, min(int(self.value - 1 + (self.value > 2)), len(descriptions) - 1))]
        else:
            description = very_low_resistances[max(0, min(int(self.value) - 51, len(very_low_resistances) - 1))]
        return f"{description} ({self.value})"


@dataclass
class MaterialResistance:
    climate: MaterialResistanceValue
    stain: MaterialResistanceValue
    acid: MaterialResistanceValue
    alkali: MaterialResistanceValue
    phosphate: MaterialResistanceValue

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(climate={self.climate}, stain={self.stain}, acid={self.acid}, alkali={self.alkali}, phosphate={self.phosphate})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(climate={repr(self.climate)}, stain={repr(self.stain)}, acid={repr(self.acid)}, alkali={repr(self.alkali)}, phosphate={repr(self.phosphate)})"


class Material:
    __wavenumber_limits: array_type
    __temperature: array_type
    __pressure: array_type

    """A class to represent a material as glass"""
    def __init__(self, name: str = "", wavenumber_limits: array_like = (0.0, np.inf),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None):
        """
        Construct a new Material object to present a meterial at a given temperature and pressure.

        :param name: The name of the material.
        :param wavenumber_limits: An array with the lowest and highest wavenumber at which this material is valid.
        :param temperature: The temperature of this material in degrees Kelvin, K.
        :param pressure: The pressure of this material in Pa = N/m^2.
        """
        self.name = name
        self.__wavenumber_limits = wavenumber_limits
        self.__temperature = asarray(0.0)
        self.temperature = temperature
        self.__pressure = asarray(0.0)
        self.pressure = pressure

    @property
    def wavenumber_limits(self) -> array_type:
        return self.__wavenumber_limits

    @wavenumber_limits.setter
    def wavenumber_limits(self, new_value: array_like):
        self.__wavenumber_limits = np.sort(asarray(new_value).real)

    @property
    def wavelength_limits(self) -> array_type:
        return 2 * np.pi / self.wavenumber_limits[::-1]

    @wavelength_limits.setter
    def wavelength_limits(self, new_value: array_like):
        self.wavenumber_limits = 2 * np.pi / asarray(new_value)[::-1]

    @property
    def angular_frequency(self) -> array_type:
        return self.wavenumber_limits * const_c

    @angular_frequency.setter
    def angular_frequency(self, new_value: array_like):
        self.wavenumber_limits = asarray(new_value) / const_c

    @property
    def temperature(self) -> array_type:
        return self.__temperature

    @temperature.setter
    def temperature(self, new_value: array_like):
        if new_value is None:
            new_value = 20.0 + 273.15
        self.__temperature = asarray(new_value)

    @property
    def pressure(self) -> array_type:
        return self.__pressure

    @pressure.setter
    def pressure(self, new_value: array_like):
        if new_value is None:
            new_value = 101.13e3
        self.__pressure = asarray(new_value)

    @property
    def refractive_index_C(self) -> array_type:
        """The refractive index at the Hydrogen Balmer series Hα C-line (deep red)."""
        return self.refractive_index(wavelength=656.281e-9)

    @property
    def refractive_index_d(self) -> array_type:
        """The refractive index at the He D3-line or d-line (Green)."""
        return self.refractive_index(wavelength=587.5618e-9)

    @property
    def refractive_index_F(self) -> array_type:
        """The refractive index at the Hydrogen Balmer series Hβ F-line (cyan)."""
        return self.refractive_index(wavelength=486.1327e-9)

    @property
    def refractive_index_g(self) -> array_type:
        """The refractive index at the Mercury g-line (blue)."""
        return self.refractive_index(wavelength=435.8343e-9)

    @property
    def constringence(self) -> array_type:
        """
        The Abbe number or Vd. High values indicate low dispersion.
        https://en.wikipedia.org/wiki/Abbe_number
        """
        return (self.refractive_index_d - 1.0) / (self.refractive_index_F - self.refractive_index_C)

    @property
    def relative_partial_dispersion_g_F(self) -> array_type:
        """
        Relative partial dispersion between the g and F lines, P_{g,F}
        https://wp.optics.arizona.edu/jgreivenkamp/wp-content/uploads/sites/11/2018/12/201-202-18-Materials.pdf
        """
        return (self.refractive_index_g - self.refractive_index_F) / (self.refractive_index_F - self.refractive_index_C)

    def permittivity(self,
                     wavenumber: Optional[array_like] = None,
                     wavelength: Optional[array_like] = None,
                     angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the permittivity for this material at the specified wavenumbers.
        Only one of wavenumber, wavelength, or angular_frequency, must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The permittivity.
        """
        return self.complex_refractive_index(wavenumber=wavenumber,
                                             wavelength=wavelength,
                                             angular_frequency=angular_frequency)**2

    def complex_refractive_index(self,
                                 wavenumber: Optional[array_like] = None,
                                 wavelength: Optional[array_like] = None,
                                 angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the complex refractive index for this material at the specified wavenumbers. Its real part is the
        conventional refractive index and its imaginary part is the extinction coefficient.
        Only one of wavenumber, wavelength, or angular_frequency, must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The complex refractive index.
        """
        return (self.refractive_index(wavenumber=wavenumber,
                                      wavelength=wavelength,
                                      angular_frequency=angular_frequency)
                + 1j * self.extinction_coefficient(wavenumber=wavenumber,
                                                   wavelength=wavelength,
                                                   angular_frequency=angular_frequency))

    def refractive_index(self,
                         wavenumber: Optional[array_like] = None,
                         wavelength: Optional[array_like] = None,
                         angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the real refractive index for this material at the specified wavenumbers, wavelengths,
        or angular_frequencies. Only one must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The refractive index, adjusted for temperature and pressure.
        """
        return self.complex_refractive_index(wavenumber=wavenumber,
                                             wavelength=wavelength,
                                             angular_frequency=angular_frequency).real

    def extinction_coefficient(self,
                               wavenumber: Optional[array_like] = None,
                               wavelength: Optional[array_like] = None,
                               angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the extinction coefficient, κ, for this material at the specified
        wavenumbers, wavelengths, or angular_frequencies. Only one must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The extinction coefficient κ, per radian
        """
        return self.complex_refractive_index(wavenumber=wavenumber,
                                             wavelength=wavelength,
                                             angular_frequency=angular_frequency).imag

    def transmittance(self, thickness: array_like = 1.0,
                      wavenumber: Optional[array_like] = None,
                      wavelength: Optional[array_like] = None,
                      angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the internal transmission, excluding Fresnel reflections, through a piece of this material with the
        specified thickness and wavenumber. Only one of wavenumber, wavelength, or angular_frequency must be specified.

        :param thickness: The distance over which the light is assumed to propagate in the material.
        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The unit-less transmission, between 0 and 1.
        """
        if wavenumber is None:
            if wavelength is None:
                wavelength = const_c * 2 * np.pi / asarray(angular_frequency)
            else:
                wavelength = asarray(wavelength)
            wavenumber = 2 * np.pi / asarray(wavelength)
        else:
            wavenumber = asarray(wavenumber)
        wavenumber = wavenumber.real

        return np.exp(-2 * wavenumber * self.extinction_coefficient(wavenumber=wavenumber) * asarray(thickness).real)

    def absorption_coefficient(self,
                               wavenumber: Optional[array_like] = None,
                               wavelength: Optional[array_like] = None,
                               angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the absorption coefficient, α, for this material at the specified
        wavenumbers, wavelengths, or angular_frequencies. Only one must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The absorption coefficient α per meter.
        """
        return self.transmittance(thickness=1.0,
                                  wavenumber=wavenumber,
                                  wavelength=wavelength,
                                  angular_frequency=angular_frequency)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Material) -> bool:
        return hash(self) == hash(other)


class Vacuum(Material):
    def __init__(self):
        super().__init__("vacuum")

    def complex_refractive_index(self,
                                 wavenumber: Optional[array_like] = None,
                                 wavelength: Optional[array_like] = None,
                                 angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the complex refractive index for this material at the specified wavenumbers, wavelengths, or
        angular_frequencies. Its real part is the conventional refractive index and its imaginary part is the extinction
        coefficient. Only one of wavenumber, wavelength, or angular_frequency, must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The complex refractive index.
        """
        if wavenumber is None:
            if wavelength is None:
                return asarray(angular_frequency) * 0.0 + 1.0
            else:
                return asarray(wavelength) * 0.0 + 1.0
        else:
            return asarray(wavenumber) * 0.0 + 1.0


class Air(Material):
    def __init__(self, temperature: Optional[array_like] = None, pressure: Optional[array_like] = None):
        """
        Construct a new Material object to present air at a given temperature and pressure.

        :param temperature: The temperature in degrees Kelvin, K.
        :param pressure: The pressure in Pa = N/m^2.
        """
        super().__init__("air", temperature=temperature, pressure=pressure)

    def complex_refractive_index(self,
                                 wavenumber: Optional[array_like] = None,
                                 wavelength: Optional[array_like] = None,
                                 angular_frequency: Optional[array_like] = None,
                                 ) -> array_type:
        """
        Calculates the complex refractive index for this material at the specified wavenumbers, wavelengths, or
        angular_frequencies. Its real part is the conventional refractive index and its imaginary part is the extinction
        coefficient. Only one of wavenumber, wavelength, or angular_frequency, must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The complex refractive index.
        """
        if wavenumber is None:
            if wavelength is None:
                wavelength = const_c * 2 * np.pi / asarray(angular_frequency)
            else:
                wavelength = asarray(wavelength)
            wavenumber = 2 * np.pi / asarray(wavelength)
        else:
            wavenumber = asarray(wavenumber)

        reference_temperature = 20.0 + 273.15
        w2 = (2 * np.pi / wavenumber)**2
        dn = 6432.8 + (2949810.0 * w2) / (146.0 * w2 - 1.0) + (25540.0 * w2) / (41.0 * w2 - 1.0)
        dn *= 1e-8

        relative_pressure = self.pressure / 101.13e3   # the pressure relative to the reference pressure, in Pa
        d_temp = self.temperature - reference_temperature  # In degrees K
        return 1.0 + relative_pressure * dn / (1.0 - d_temp * 3.4785e-3)


class MaterialLibrary:
    name: str = ""
    description: str = ""
    materials: Sequence[Material] = []

    def __init__(self, name: str, description: str = "", materials: Sequence[Material] = tuple()):
        self.name = name
        self.description = description
        self.materials = materials

    @property
    def names(self) -> Sequence[str]:
        return [_.name for _ in self.materials]

    def __len__(self) -> int:
        return len(self.materials)

    def __contains__(self, item: str | Material) -> bool:
        return item in (self.materials if isinstance(item, Material) else self.names)

    def __getitem__(self, item: int | str) -> Optional[Material]:
        """
        Get a `Material` by integer index or by case-sensitive name.

        :param item: An integer index or a string that indicates the name of the material.

        :return: The first material that matches the description. None if material not found.
        """
        if isinstance(item, int):
            if 0 <= item < len(self.materials):
                return self.materials[item]
            else:
                raise IndexError(f"Invalid index {item}, it should be in [0, {len(self.materials)}) for this material library.")
        elif isinstance(item, str):
            for _ in self.materials:
                if _.name == item:
                    return _
        else:
            raise TypeError(f"The index for {self.__class__.__name__}[{item}] must be of type int or str, not {item.__class__}")
        return None

    def __iter__(self) -> Iterator[Material]:
        return self.materials.__iter__()

    def find_all(self, name_pattern: str | re.Pattern | None,
                 wavenumber: array_like = (0.0, np.inf),
                 wavelength: array_like = (np.inf, 0.0),
                 angular_frequency: array_like = (0.0, np.inf)) -> Sequence[Material]:
        """
        Lists all matching `Material`s.

        :param name_pattern: An optional sub-string or regular expression that matches the name of the material.
        :param wavenumber: The wavenumber range that all materials must be able to handle.
        :param wavelength: The wavelength range that all materials must be able to handle.
        :param angular_frequency: The angular_frequency range that all materials must be able to handle.

        :return: A sequence of all materials that match the description.
        """
        if wavenumber is None:
            if wavelength is None:
                wavelength = const_c * 2 * np.pi / asarray(angular_frequency)
            else:
                wavelength = asarray(wavelength)
            wavenumber = 2 * np.pi / wavelength
        else:
            wavenumber = asarray(wavenumber)

        if wavenumber.size == 1:
            wavenumber = asarray([wavenumber, wavenumber])
        else:
            wavenumber.sort()

        if isinstance(name_pattern, str) and name_pattern[0] == '/':
            _, pattern, flags_str = name_pattern.split('/')
            flags = re.NOFLAG
            if 'i' in flags_str:
                flags |= re.IGNORECASE
            if 'm' in flags_str:
                flags |= re.MULTILINE
            name_pattern = re.compile(pattern, flags=flags)

        # Find all materials that can handle the requested wavenumber range
        matching_materials = [_ for _ in self.materials
                              if _.wavenumber_limits[0] <= wavenumber[0] and wavenumber[-1] <= _.wavenumber_limits[-1]]

        # Find all materials with a matching name
        if isinstance(name_pattern, str):
            return [_ for _ in matching_materials if name_pattern.lower() in _.name.lower()]
        if isinstance(name_pattern, re.Pattern):
            return [_ for _ in matching_materials if name_pattern.match(_.name) is not None]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.description}, {repr(self.materials)})"


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


