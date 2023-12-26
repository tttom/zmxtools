from __future__ import  annotations

import re
import typing
from typing import Optional, Self, List, Sequence, Iterator, Callable
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


BinaryFileLike = BytesFile | typing.BinaryIO
FileLike = BinaryFileLike | typing.IO
PathLike = os.PathLike | str


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


class FunctionMaterial(Material):
    """A class to represent a material as glass"""
    def __init__(self, name: str = "", wavenumber_limits: array_like = (0.0, np.inf),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
                 permittivity_function: Optional[Callable[[array_type, array_type, array_type], array_type]] = None):
        """
        Construct a new Material object to present a meterial at a given temperature and pressure.

        :param name: The name of the material.
        :param wavenumber_limits: An array with the lowest and highest wavenumber at which this material is valid.
        :param temperature: The temperature of this material in degrees Kelvin, K.
        :param pressure: The pressure of this material in Pa = N/m^2.
        :param permittivity_function: A function that returns an array of refractive indices at the specified
            wavenumbers, temperatures, and pressures. The returned array should be broadcasted according to its input
            arguments.
        """
        super().__init__(name=name, wavenumber_limits=wavenumber_limits, temperature=temperature, pressure=pressure)
        self.__permittivity_function = permittivity_function

    def permittivity(self,
                     wavenumber: Optional[array_like] = None,
                     wavelength: Optional[array_like] = None,
                     angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the relative permittivity for this material at the specified wavenumbers, wavelengths, or angular
        frequencies. Only one must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The relative permittivity.
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
        return self.__permittivity_function(wavenumber, self.temperature, self.pressure)

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
        return self.permittivity(wavenumber=wavenumber, wavelength=wavelength, angular_frequency=angular_frequency) ** 0.5


class Vacuum(FunctionMaterial):
    def __init__(self):
        super().__init__("vacuum", permittivity_function=lambda k, t, p: asarray(k+t+p) * 0.0 + 1.0)


class CiddorAir(FunctionMaterial):
    def __init__(self, name: str = "air", wavenumber_limits: array_like = (2 * np.pi / 1700e-3, 2 * np.pi / 300e-3),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
                 relative_humidity: Optional[array_like] = None, co2_mole_fraction: Optional[array_like] = None):
        """
        Construct a new Material object to present air at a given temperature, pressure, humidty, and CO2 concentration.
        The Ciddor formula is used: https://emtoolbox.nist.gov/Wavelength/Documentation.asp#AppendixAIII
        https://emtoolbox.nist.gov/Wavelength/Ciddor.asp

        :param temperature: The temperature in degrees Kelvin, K.
        :param pressure: The pressure in Pa = N/m^2.
        :param relative_humidity: between 0 and 1.
        :param co2_mole_fraction: in mole per mole of air.
        """
        self.relative_humidity = relative_humidity if relative_humidity is not None else 0.50
        self.co2_mole_fraction = co2_mole_fraction if co2_mole_fraction is not None else 450e-6

        def permittivity_function(wavenumber: array_type, temperature: array_type, pressure: array_type) -> array_type:
            # https://emtoolbox.nist.gov/Wavelength/Documentation.asp#AppendixAIII

            w = [295.235, 2.6422, -0.03238, 0.004028]
            k = [238.0185, 5792105, 57.362, 167917]
            a = [1.58123e-6, -2.9331e-8, 1.1043e-10]
            b = [5.707e-6, -2.051e-8]
            c = [1.9898e-4, -2.376e-6]
            d = 1.83e-11
            e = -0.765e-8
            pressure_reference = 101325
            temperature_reference = 288.15
            Z_a = 0.9995922115
            rho_vs = 0.00985938
            R = 8.314472
            M_v = 0.018015

            wavelength = 2 * np.pi / wavenumber
            spatial_frequency_um_sqd = 1 / (wavelength / 1e-6)**2
            r_as = 1e-8 * (k[1] / (k[0] - spatial_frequency_um_sqd) + k[3] / (k[2] - spatial_frequency_um_sqd))
            r_vs = 1.022e-8 * (w[0] + w[1] * spatial_frequency_um_sqd + w[2] * spatial_frequency_um_sqd**2 + w[3] * spatial_frequency_um_sqd**3)
            M_a = 0.0289635 + 1.2011 * 1e-8 * (co2_mole_fraction / 1e-6 - 400)
            r_axs = r_as * (1 + 5.34 * 1e-7 * (co2_mole_fraction / 1e-6 - 450))
            temperature_celcius = temperature - 273.15

            abg = [1.00062, 3.14e-8, 5.60e-7]
            f_pt = abg[0] + abg[1] * pressure + abg[2] * temperature_celcius**2

            over_water = temperature > 273.16
            # For saturation vapor pressure over water
            omega = temperature - 2.38555575678e-01 / (temperature - 6.50175348448e+02)
            p_sv_a = omega**2 + 1.16705214528e+03 * omega - 7.24213167032e+05
            p_sv_b = -1.70738469401e+01 * omega**2 + 1.20208247025e+04 * omega - 3.23255503223e+06
            p_sv_c = 1.49151086135e+01 * omega**2 - 4.82326573616e+03 * omega + 4.05113405421e+05
            p_sv_x = -p_sv_b + (p_sv_b**2 - 4 * p_sv_a * p_sv_c)**0.5
            p_sv_water = 1e6 * (2 * p_sv_c / p_sv_x)**4

            # For saturation vapor pressure over ice
            theta = temperature / 273.16
            p_sv_y = -13.928169 * (1 - theta**-1.5) + 34.7078238 * (1 - theta**-1.25)
            p_sv_ice = 611.657 * np.exp(p_sv_y)

            p_sv = p_sv_water * over_water + p_sv_ice * (1 - over_water)

            x_v = relative_humidity * f_pt * p_sv / pressure

            z_m = 1 - (pressure / temperature) * (a[0] + a[1] * temperature_celcius +
                                                  a[2] * temperature_celcius**2 +
                                                  (b[0] + b[1] * temperature_celcius) * x_v +
                                                  (c[0] + c[1] * temperature_celcius) * x_v**2) \
                  + (pressure / temperature) ** 2 * (d + e * x_v**2)
            rho_axs = pressure_reference * M_a / (Z_a * R * temperature_reference)
            rho_v = x_v * pressure * M_v / (z_m * R * temperature)
            rho_a = (1 - x_v) * pressure * M_a / (z_m * R * temperature)

            n = 1.0 + (rho_a / rho_axs) * r_axs + (rho_v / rho_vs) * r_vs

            return n ** 2

        super().__init__(name=name, wavenumber_limits=wavenumber_limits,
                         temperature=temperature, pressure=pressure, permittivity_function=permittivity_function)


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


