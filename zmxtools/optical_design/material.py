from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from zmxtools.optical_design import log
from zmxtools.utils import const_c
from zmxtools.utils.array import array_like, array_type, asarray

__all__ = ['log', 'CiddorAir', 'FunctionMaterial', 'Material', 'MaterialLibrary', 'MaterialResistance',
           'PolynomialMaterial', 'SimpleAir', 'Vacuum', 'VACUUM',
           ]

log = log.getChild(__name__)


@dataclass
class MaterialResistanceValue:
    """A class to represent the physical resistance of a material on a numerical scale."""

    value: float = 2
    limits: Sequence[float] = (0, 5)

    def __float__(self) -> float:
        """The material resistance value as a floating point number."""
        return float(self.value)

    def __int__(self) -> int:
        """The material resistance value as an integer number."""
        return int(self.value)

    def __str__(self) -> str:
        """Formats this resistance value as a string for display."""
        descriptions = ['high', 'medium-high', 'medium', 'medium-low', 'low']
        very_low_resistances = ['very-low', 'very-low', 'very-very-low', 'extremely-low', 'too-low']
        if min(self.limits) <= self.value <= 25:
            if self.limits[0] > 0:
                description = descriptions[max(0, min(int(self.value) - 1, len(descriptions) - 1))]
            else:
                description = descriptions[max(0, min(int(self.value - 1 + (self.value > 2)), len(descriptions) - 1))]
        else:
            description = very_low_resistances[max(0, min(int(self.value) - 51, len(very_low_resistances) - 1))]
        return f'{description} ({self.value})'


@dataclass
class MaterialResistance:
    """A data class to represent the resistance of a material to various physical factors."""

    climate: MaterialResistanceValue
    stain: MaterialResistanceValue
    acid: MaterialResistanceValue
    alkali: MaterialResistanceValue
    phosphate: MaterialResistanceValue

    def __str__(self) -> str:
        """Formats this resistance as a string for display."""
        return (f'{self.__class__.__name__}(climate={self.climate}, stain={self.stain},' +
                f' acid={self.acid}, alkali={self.alkali}, phosphate={self.phosphate})'
                )

    def __repr__(self) -> str:
        """The representation of this object as a string."""
        return (f'{self.__class__.__name__}(climate={repr(self.climate)}, stain={repr(self.stain)},' +
                f'  acid={repr(self.acid)}, alkali={repr(self.alkali)}, phosphate={repr(self.phosphate)})'
                )


class Material:
    """A class to represent a material as glass."""

    __wavenumber_limits: array_type
    __temperature: array_type
    __pressure: array_type

    def __init__(self, name: str = '', wavenumber_limits: array_like = (0, np.inf),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
                 ):
        """
        Construct a new Material object to present a material at a given temperature and pressure.

        :param name: The name of the material.
        :param wavenumber_limits: An array with the lowest and highest wavenumber at which this material is valid.
        :param temperature: The temperature of this material in degrees Kelvin, K (default 20+273.15 K).
        :param pressure: The pressure of this material in Pa = N/m^2 (default 101.13 kPa = 1 atm)
        """
        self.name = name
        self.__wavenumber_limits = wavenumber_limits
        self.__temperature = asarray(0)
        self.temperature = temperature
        self.__pressure = asarray(0)
        self.pressure = pressure

    @property
    def wavenumber_limits(self) -> array_type:
        """The lower and upper bound of the validity of this representation."""
        return self.__wavenumber_limits

    @wavenumber_limits.setter
    def wavenumber_limits(self, new_value: array_like):
        self.__wavenumber_limits = np.sort(asarray(new_value).real)

    @property
    def wavelength_limits(self) -> array_type:
        """The lower and upper bound of the validity of this representation."""
        return 2 * np.pi / self.wavenumber_limits[::-1]

    @wavelength_limits.setter
    def wavelength_limits(self, new_value: array_like):
        self.wavenumber_limits = 2 * np.pi / asarray(new_value)[::-1]

    @property
    def angular_frequency_limits(self) -> array_type:
        """The lower and upper bound of the validity of this representation."""
        return self.wavenumber_limits * const_c

    @angular_frequency_limits.setter
    def angular_frequency_limits(self, new_value: array_like):
        self.wavenumber_limits = asarray(new_value) / const_c

    @property
    def temperature(self) -> array_type:
        """The temperature that this material is at."""
        return self.__temperature

    @temperature.setter
    def temperature(self, new_value: array_like):
        if new_value is None:
            new_value = 20.0 + 273.15
        self.__temperature = asarray(new_value).real

    @property
    def pressure(self) -> array_type:
        """The pressure that this material is at."""
        return self.__pressure

    @pressure.setter
    def pressure(self, new_value: array_like):
        if new_value is None:
            new_value = 101.13e3  # in Pa, 1 atm
        self.__pressure = asarray(new_value).real

    @property
    def refractive_index_c(self) -> array_type:
        """The refractive index at the Hydrogen Balmer series Hα C-line (deep red)."""
        return self.refractive_index(wavelength=656.281e-9)

    @property
    def refractive_index_d(self) -> array_type:
        """The refractive index at the He D3-line or d-line (Green)."""
        return self.refractive_index(wavelength=587.5618e-9)

    @property
    def refractive_index_f(self) -> array_type:
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
        return (self.refractive_index_d - 1.0) / (self.refractive_index_f - self.refractive_index_c)

    @property
    def relative_partial_dispersion_g_f(self) -> array_type:
        """
        Relative partial dispersion between the g and F lines, P_{g,F}.

        https://wp.optics.arizona.edu/jgreivenkamp/wp-content/uploads/sites/11/2018/12/201-202-18-Materials.pdf
        """
        return (self.refractive_index_g - self.refractive_index_f) / (self.refractive_index_f - self.refractive_index_c)

    def permittivity(self,
                     wavenumber: Optional[array_like] = None,
                     wavelength: Optional[array_like] = None,
                     angular_frequency: Optional[array_like] = None,
                     ) -> array_type:
        """
        Calculates the relative permittivity for this material at the specified wavenumbers.

        Only one of wavenumber, wavelength, or angular_frequency, must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The relative permittivity.
        """
        return self.complex_refractive_index(wavenumber=wavenumber,
                                             wavelength=wavelength,
                                             angular_frequency=angular_frequency,
                                             ) ** 2

    def complex_refractive_index(self,
                                 wavenumber: Optional[array_like] = None,
                                 wavelength: Optional[array_like] = None,
                                 angular_frequency: Optional[array_like] = None,
                                 ) -> array_type:
        """
        Calculates the complex refractive index for this material at the specified wavenumbers.

        Its real part is the conventional refractive index and its imaginary part is the extinction coefficient.
        Only one of wavenumber, wavelength, or angular_frequency, must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The complex refractive index.
        """
        return (self.refractive_index(
            wavenumber=wavenumber, wavelength=wavelength, angular_frequency=angular_frequency,
        ) + 1j * self.extinction_coefficient(
            wavenumber=wavenumber, wavelength=wavelength, angular_frequency=angular_frequency,
        ))

    def refractive_index(self,
                         wavenumber: Optional[array_like] = None,
                         wavelength: Optional[array_like] = None,
                         angular_frequency: Optional[array_like] = None,
                         ) -> array_type:
        """
        Calculates the real refractive index for this material at the specified wavenumbers.

        Alternatively to wavenumber, also wavelength, or angular_frequency may be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The refractive index, adjusted for temperature and pressure.
        """
        return self.complex_refractive_index(wavenumber=wavenumber,
                                             wavelength=wavelength,
                                             angular_frequency=angular_frequency,
                                             ).real

    def extinction_coefficient(self,
                               wavenumber: Optional[array_like] = None,
                               wavelength: Optional[array_like] = None,
                               angular_frequency: Optional[array_like] = None,
                               ) -> array_type:
        """
        Calculates the extinction coefficient, κ, for this material at the specified wavenumbers.

        Instead of wavenumbers, also wavelengths, or angular_frequencies can be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The extinction coefficient κ, per radian
        """
        return self.complex_refractive_index(wavenumber=wavenumber,
                                             wavelength=wavelength,
                                             angular_frequency=angular_frequency,
                                             ).imag

    def transmittance(self, thickness: array_like = 1.0,
                      wavenumber: Optional[array_like] = None,
                      wavelength: Optional[array_like] = None,
                      angular_frequency: Optional[array_like] = None,
                      ) -> array_type:
        """
        Calculates the internal transmission, excluding Fresnel reflections, through a piece of this material.

        Only one of wavenumber, wavelength, or angular_frequency must be specified.

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
                               angular_frequency: Optional[array_like] = None,
                               ) -> array_type:
        """
        Calculates the absorption coefficient, α, for this material at the specified wavenumbers.

        Instead of wavenumbers, also wavelengths, or angular_frequencies can be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The absorption coefficient α per meter.
        """
        return self.transmittance(thickness=1.0,
                                  wavenumber=wavenumber,
                                  wavelength=wavelength,
                                  angular_frequency=angular_frequency,
                                  )

    def __repr__(self) -> str:
        """
        The representation of this object as a string.

        This must be a full, unique description of this object, sufficient to permit its reconstruction.
        If not, the hash and equality magic methods will not work as expected.
        """
        return f'{self.__class__.__name__}({self.name})'

    def __hash__(self) -> int:
        """The numerical hash of this object. This is, for instance, used when indexing in a dictionary."""
        return hash(repr(self))

    def __eq__(self, other: Material) -> bool:
        """As a fall-back, comparison of one object with another is done based on its hash."""
        return hash(self) == hash(other)


class FunctionMaterial(Material):
    """A class to represent a material as glass."""

    def __init__(
        self, name: str = '', wavenumber_limits: array_like = (0, np.inf),
        temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
        permittivity_function: Optional[Callable[[array_type, array_type, array_type], array_type]] = None,
        complex_refractive_index_function: Optional[Callable[[array_type, array_type, array_type], array_type]] = None,
    ):
        """
        Construct a new Material object to present a material at a given temperature and pressure.

        :param name: The name of the material.
        :param wavenumber_limits: An array with the lowest and highest wavenumber at which this material is valid.
        :param temperature: The temperature of this material in degrees Kelvin, K.
        :param pressure: The pressure of this material in Pa = N/m^2.
        :param permittivity_function: A function that returns an array of relative permittivities at the
            specified wavenumbers, temperatures, and pressures. The returned array should be broadcasted according to
            its input arguments.
        :param complex_refractive_index_function: A function that returns an array of refractive indices at the
            specified wavenumbers, temperatures, and pressures. The returned array should be broadcasted according to
            its input arguments.
        """
        if permittivity_function is None and complex_refractive_index_function is None:
            raise ValueError(
                'Either the permittivity function of the refractive index function must be specified, not both.',
            )
        super().__init__(name=name, wavenumber_limits=wavenumber_limits, temperature=temperature, pressure=pressure)
        if permittivity_function is None:
            self.__permittivity_function = lambda k, t, p: complex_refractive_index_function(k, t, p) ** 2
        else:
            self.__permittivity_function = permittivity_function
        self.__complex_refractive_index_function = (
            complex_refractive_index_function
        ) if complex_refractive_index_function is not None else lambda k, t, p: permittivity_function(k, t, p) ** 0.5

    def permittivity(self,
                     wavenumber: Optional[array_like] = None,
                     wavelength: Optional[array_like] = None,
                     angular_frequency: Optional[array_like] = None,
                     ) -> array_type:
        """
        Calculates the relative permittivity for this material at the specified wavenumbers.

        Instead of wavenumber, also wavelength, or angular frequency can be specified.

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
                                 angular_frequency: Optional[array_like] = None,
                                 ) -> array_type:
        """
        Calculates the complex refractive index for this material at the specified wavenumbers.

        Its real part is the conventional refractive index and its imaginary part is the extinction coefficient.
        Only one of wavenumber, wavelength, or angular_frequency, must be specified.

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
        wavenumber = wavenumber.real
        return self.__complex_refractive_index_function(wavenumber, self.temperature, self.pressure)


AdjustRIType = Callable[[Callable[[array_type], array_type], array_type, array_type, array_type], array_type]


class ModelGlassMaterial(FunctionMaterial):
    """
    A class to represent a model glass with a specified refractive index at 587.5618nm and constringence (Abbe number).

    The constringence is defined as (nd - 1) / (nF - nC): https://en.wikipedia.org/wiki/Abbe_number
    where
    * nF is the refractive index at 486.1327nm, the blue hydrogen line, F
    * nd is the refractive index at 587.5618nm, the yellow helium line, d
    * nC is the refractive index at 656.2725nm, the red hydrogen line, C

    This material perfectly interpolates and extrapolates the refractive index for all wavelengths.
    """
    def __init__(self, name: str = '', refractive_index: float = 1, constringence: float = np.nan):
        """
        Constructs a model glass from just the refractive index and optionally the Abbe number or constringence.

        This function is based on the code in:
        https://github.com/mjhoptics/opticalglass/blob/master/src/opticalglass/buchdahl.py#L168

        :param name: The glass name.
        :param refractive_index: The refractive index at the central d-line of 587.5618nm.
        :param constringence: The constringence or Abbe number. If not specified, a constant refractive index is used.
        """
        # Modeled using a 6-digit glass specification:
        b = -0.064667
        m = -1.604048

        long_wavelength = 656.2725e-9  # red hydrogen line, C
        center_wavelength = 587.5618e-9  # yellow helium line, d
        short_wavelength = 486.1327e-9  # blue hydrogen line, F

        def buchdahl_chromatic_coordinate(wavelength_difference: array_like) -> array_type:
            """Calculate the Buchdahl chromatic coordinate."""
            return 1 / (2.5 + 1e-6 / asarray(wavelength_difference))

        omega_long = buchdahl_chromatic_coordinate(long_wavelength - center_wavelength)  # red hydrogen line, C
        omega_short = buchdahl_chromatic_coordinate(short_wavelength - center_wavelength)  # blue hydrogen line, F

        delta_omega = omega_short - omega_long
        delta_omega_2 = omega_short ** 2 - omega_long ** 2

        def complex_refractive_index_function(wavenumber: array_like, t: array_like, p: array_like) -> array_type:
            wavenumber = asarray(wavenumber)

            if not np.isnan(constringence) and constringence != 0:
                # Fit the curve
                dFC = (refractive_index - 1) / constringence
                v2 = (dFC - b * delta_omega) / (m * delta_omega - delta_omega_2)

                omega = buchdahl_chromatic_coordinate(2 * np.pi / wavenumber - center_wavelength)

                return refractive_index + (b + m * v2) * omega + v2 * omega ** 2

            return np.full(wavenumber.shape, refractive_index)

        super().__init__(name=name, wavenumber_limits=(0, np.inf),
                         complex_refractive_index_function=complex_refractive_index_function,
                         )


class PolynomialMaterial(FunctionMaterial):
    """A class to represent materials with a refractive index distribution that is described by a polynomial."""

    def __init__(self, name: str = '', wavenumber_limits: array_like = (0, np.inf),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
                 refractive_index_model: bool = False,
                 factors_um: array_like = (), exponents: array_like = (), poles_um2: array_like = (),
                 adjust_refractive_index_function: AdjustRIType = lambda n, t, p: n,
                 ):
        r"""
        Construct a material with a relative permittivity or refractive index that is given a rational polynomial.

        The rational polynomial can be written as:

        .. math::
            \epsilon_r(\lambda) or n(\lambda) = \sum_{i=0} c_i \frac{\lambda^e}{\lambda^2 - p_i},

        where :math:`c_i` are the coefficient factors and :math:`p_i` are the poles, while `\lambda^2` is the squared
        wavelength in micrometers, and :math:`e` are the exponents for the wavelengths.

        :param name: The name of this material.
        :param wavenumber_limits: The wavelength limits of validity.
        :param temperature: The temperature this material is at in degrees K.
        :param pressure: The pressure this material is at (in Pa).
        :param refractive_index_model: Set to True to model the material's refractive index as a rational polynomial
            instead of its permittivity.
        :param factors_um: The coefficient factors for a polynomial in the micrometer-wavelengths
        :param exponents: The exponents for the wavelength in the numerator of each term.
        :param poles_um2: The poles in the polynomial terms in micrometer^2. Set to NaN to not divide the term.
        :param adjust_refractive_index_function: A function that returns the complex refractive index as a function of
            the reference refractive index function, the wavenumber, temperature, and pressure.
        """
        self.refractive_index_model = refractive_index_model
        self.factors_um = asarray(factors_um)
        self.exponents = asarray([
            *exponents,
            *(0 for _ in range(len(self.factors_um) - len(exponents))),
        ])  # 0-pad to same length
        self.poles_um2 = asarray([
            *poles_um2,
            *(np.nan for _ in range(len(self.factors_um) - len(poles_um2))),
        ])  # NaN-pad to same length

        def formula(wavenumber: array_type) -> array_type:
            """
            The permittivity or refractive index (when refractive_index_model == True) as a function of wavenumber.

            The values are given at the reference temperature and pressure.
            """
            wavelength_um = 2 * np.pi / wavenumber / 1e-6
            wavelength_um2 = wavelength_um ** 2

            result = 0
            for factor_um, exponent, pole_um2 in zip(self.factors_um, exponents, self.poles_um2):
                if factor_um != 0:
                    term = factor_um
                    if exponent != 0:
                        term = term * wavelength_um ** exponent
                    if not np.isnan(pole_um2):
                        term = term / (wavelength_um2 - pole_um2)
                    result = result + term
            return result

        # Convert formula to refractive index equivalent
        reference_refractive_index_function = formula if self.refractive_index_model else (
            lambda _: formula(_) ** 0.5
        )

        super().__init__(name=name, wavenumber_limits=wavenumber_limits, temperature=temperature, pressure=pressure,
                         complex_refractive_index_function=lambda k, t, p: adjust_refractive_index_function(
                             reference_refractive_index_function, k, t, p,
                         ),
                         )

    def __repr__(self) -> str:
        """The representation of this object as a string."""
        return (f'{self.__class__.__name__}({self.name}, factors_um={self.factors_um},' +
                f'  exponents={self.exponents}, poles_um2={self.poles_um2})'
                )


class Vacuum(FunctionMaterial):
    """A class for the vacuum material object."""

    def __init__(self):
        """Construct the vacuum."""
        super().__init__('vacuum', complex_refractive_index_function=lambda k, t, p: asarray(np.ones_like(k + t + p)))


VACUUM = Vacuum()


class CiddorAir(FunctionMaterial):
    """A class to represent a glass that is represented by th Ciddor formula."""

    def __init__(self, name: str = 'air', wavenumber_limits: array_like = (2 * np.pi / 1700e-3, 2 * np.pi / 300e-3),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
                 relative_humidity: Optional[array_like] = None, co2_mole_fraction: Optional[array_like] = None,
                 ):
        """
        Construct a new Material object to present air at a given temperature, pressure, humidity, and CO2 level.

        The Ciddor formula is used: https://emtoolbox.nist.gov/Wavelength/Documentation.asp#AppendixAIII
        https://emtoolbox.nist.gov/Wavelength/Ciddor.asp
        This typically yields 9 significant digits. If 7 digits is sufficient,
        consider using :py:class:``SimpleAir` instead.

        :param temperature: The temperature in degrees Kelvin, K.
        :param pressure: The pressure in Pa = N/m^2.
        :param relative_humidity: between 0 and 1.
        :param co2_mole_fraction: in mole per mole of air.
        """
        self.relative_humidity = relative_humidity if relative_humidity is not None else 0.5
        self.co2_mole_fraction = co2_mole_fraction if co2_mole_fraction is not None else 450e-6

        def refractive_index_function(wavenumber: array_type, t: array_type, p: array_type,
                                      ) -> array_type:
            # https://emtoolbox.nist.gov/Wavelength/Documentation.asp#AppendixAIII

            w = [295.235, 2.6422, -0.03238, 0.004028]
            k = [238.0185, 5792105, 57.362, 167917]
            a = [1.58123e-6, -2.9331e-8, 1.1043e-10]
            b = [5.707e-6, -2.051e-8]
            c = [1.9898e-4, -2.376e-6]
            d = 1.83e-11
            e = -0.765e-8
            pressure_reference = 101325  # in Pa, 1 atm
            temperature_reference = 288.15
            z_a = 0.9995922115
            rho_vs = 0.00985938
            r_gas_const = 8.314472
            m_v = 0.018015

            wavelength = 2 * np.pi / wavenumber
            spatial_frequency_um_sqd = 1 / (wavelength / 1e-6)**2
            r_as = 1e-8 * (k[1] / (k[0] - spatial_frequency_um_sqd) + k[3] / (k[2] - spatial_frequency_um_sqd))
            r_vs = 1.022e-8 * sum(v * spatial_frequency_um_sqd ** _ for _, v in enumerate(w[:4]))
            m_a = 0.0289635 + 1.2011 * 1e-8 * (co2_mole_fraction / 1e-6 - 400)
            r_axs = r_as * (1 + 5.34 * 1e-7 * (co2_mole_fraction / 1e-6 - 450))
            temperature_celcius = t - 273.15

            abg = 1.00062, 3.14e-8, 5.60e-7
            f_pt = abg[0] + abg[1] * p + abg[2] * temperature_celcius ** 2

            over_water = t > 273.16
            # For saturation vapor pressure over water
            omega = t - 0.238555575678 / (t - 650.175348448)
            p_sv_a = omega**2 + 1167.05214528 * omega - 724213.167032
            p_sv_b = -17.0738469401 * omega**2 + 12020.8247025 * omega - 3232555.03223
            p_sv_c = 14.9151086135 * omega**2 - 4823.26573616 * omega + 405113.405421
            p_sv_x = -p_sv_b + (p_sv_b**2 - 4 * p_sv_a * p_sv_c)**0.5
            p_sv_water = 1e6 * (2 * p_sv_c / p_sv_x)**4

            # For saturation vapor pressure over ice
            theta = t / 273.16
            p_sv_y = -13.928169 * (1 - theta**-1.5) + 34.7078238 * (1 - theta**-1.25)
            p_sv_ice = 611.657 * np.exp(p_sv_y)

            p_sv = p_sv_water * over_water + p_sv_ice * (1 - over_water)

            x_v = relative_humidity * f_pt * p_sv / (p + (p == 0))

            z_m = (1 - (p / t) * (a[0] + a[1] * temperature_celcius +
                                  a[2] * temperature_celcius ** 2 +
                                  (b[0] + b[1] * temperature_celcius) * x_v +
                                  (c[0] + c[1] * temperature_celcius) * x_v ** 2
                                  ) +
                   (p / t) ** 2 * (d + e * x_v ** 2)
                   )
            rho_axs = pressure_reference * m_a / (z_a * r_gas_const * temperature_reference)
            rho_v = x_v * p * m_v / (z_m * r_gas_const * t)
            rho_a = (1 - x_v) * p * m_a / (z_m * r_gas_const * t)

            return 1.0 + (rho_a / rho_axs) * r_axs + (rho_v / rho_vs) * r_vs

        super().__init__(name=name, wavenumber_limits=wavenumber_limits,
                         temperature=temperature, pressure=pressure,
                         complex_refractive_index_function=refractive_index_function,
                         )


class SimpleAir(FunctionMaterial):
    """A class to represent air in a relatively simple way."""

    def __init__(self, name: str = 'air', wavenumber_limits: array_like = (2 * np.pi / 1700e-3, 2 * np.pi / 300e-3),
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None,
                 ):
        """
        Construct a new Material object to present air at a given temperature and pressure.

        This simplified formula typically yields 7 significant digits if the relative humidity is 0%, and 5 otherwise.
        Use the CiddorAir to handle relative humidity and CO2 concentration up to 9 significant digits.

        :param temperature: The temperature in degrees Kelvin, K.
        :param pressure: The pressure in Pa = N/m^2.
        """
        def refractive_index_function(wavenumber: array_type, t: array_type, p: array_type,
                                      ) -> array_type:
            wavelength = 2 * np.pi / wavenumber
            spatial_frequency_um_sqd = 1 / (wavelength / 1e-6)**2
            pressure_reference = 101325  # 1 atm
            temperature_celcius = t - 273.15

            n_ref = 1.0 + (6432.8 +
                           2949810 / (146 - spatial_frequency_um_sqd) +
                           25540 / (41 - spatial_frequency_um_sqd)
                           ) * 1e-8
            return 1.0 + (n_ref - 1.0) * p / pressure_reference / (1.0 + (temperature_celcius - 15) * 3.4785e-3)

        super().__init__(name=name, wavenumber_limits=wavenumber_limits,
                         temperature=temperature, pressure=pressure,
                         complex_refractive_index_function=refractive_index_function,
                         )


class MaterialLibrary:
    """A class to represent a collection of optical materials."""

    def __init__(self, name: str, description: str = '', materials: Sequence[Material] = ()):
        """
        Construct a new material catalog from a sequence of `Material` objects.

        :param name: The catalog name.
        :param description: An optional description of this collection.
        :param materials: The sequence of materials to store in this material collection.
        """
        self.name: str = name
        self.description: str = description
        self.materials: Sequence[Material] = materials

    @property
    def names(self) -> Sequence[str]:
        """All the material names in their listed order."""
        return [_.name for _ in self.materials]

    def __len__(self) -> int:
        """The number of materials in this library."""
        return len(self.materials)

    def __contains__(self, item: str | Material) -> bool:
        """
        Returns true if a material or a material with the specified name is found in this is library.

        This is used with the :py:func:``in` keyword.

        :param item: The material or material name.

        :return: True if found, False otherwise.
        """
        if isinstance(item, Material):
            return item in self.materials
        if isinstance(item, str) and item[0] == '/':  # Interpret as regular expression
            _, pattern, flags_str = item.split('/')
            flags = re.NOFLAG
            if 'i' in flags_str:
                flags |= re.IGNORECASE
            if 'm' in flags_str:
                flags |= re.MULTILINE
            item = re.compile(pattern, flags=flags)
        if isinstance(item, str):
            for n in self.names:
                if item.lower() == n.lower():
                    return True
            return False
        for name in self.names:
            if item.match(name) is not None:
                return True
        return False

    def __getitem__(self, item: int | str) -> Optional[Material]:
        """
        Get a `Material` by integer index or by case-sensitive name.

        :param item: An integer index or a string that indicates the name of the material.

        :return: The first material that matches the description. None if material not found.
        """
        if isinstance(item, int):
            if 0 <= item < len(self.materials):
                return self.materials[item]
            raise IndexError(
                f'Invalid index {item}, it should be in [0, {len(self.materials)}) for this material library.',
            )
        elif isinstance(item, str):
            for _ in self.materials:
                if _.name == item:
                    return _
        else:
            raise TypeError(
                f'The index for {self.__class__.__name__}[{item}] must be of type int or str, not {item.__class__}',
            )
        return None

    def __iter__(self) -> Iterator[Material]:
        """Returns an iterator over all materials in this library. This is used in a for-loop."""
        return self.materials.__iter__()

    def find_all(self, name_pattern: str | re.Pattern | None,
                 wavenumber: array_like = (np.inf, 0),
                 wavelength: array_like = (0, np.inf),
                 angular_frequency: array_like = (np.inf, 0),
                 ) -> Sequence[Material]:
        """
        Lists all matching materials.

        :param name_pattern: An optional sub-string or regular expression that matches the name of the material.
        :param wavenumber: The wavenumber range that all materials must be able to handle. Default: match any.
        :param wavelength: The wavelength range that all materials must be able to handle. Default: match any.
        :param angular_frequency: The angular_frequency range that all materials must be able to handle.
            Default: match any.

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

        if isinstance(name_pattern, str) and name_pattern[0] == '/':  # Interpret as regular expression
            _, pattern, flags_str = name_pattern.split('/')
            flags = re.NOFLAG
            if 'i' in flags_str:
                flags |= re.IGNORECASE
            if 'm' in flags_str:
                flags |= re.MULTILINE
            name_pattern = re.compile(pattern, flags=flags)
        matching_materials = self.materials

        # Find all materials with a matching name
        if isinstance(name_pattern, str):
            matching_materials = [_ for _ in matching_materials if name_pattern in _.name]
        if isinstance(name_pattern, re.Pattern):
            matching_materials = [_ for _ in matching_materials if name_pattern.match(_.name) is not None]

        # Find all materials that can handle the requested wavenumber range
        return [_
                for _ in matching_materials
                if _.wavenumber_limits[0] <= wavenumber[0] and wavenumber[-1] <= _.wavenumber_limits[-1]
                ]

    def __repr__(self) -> str:
        """The representation of this object as a string."""
        return f'{self.__class__.__name__}({self.name}, {self.description}, {repr(self.materials)})'
