"""
Parse Zemax Glass AGF files.

The following projects have been useful references:

 - RefractiveIndex.info, https://github.com/polyanskiy/refractiveindex.info-database

 - The ZemaxGlass project, https://github.com/nzhagen/zemaxglass

 - RayOpt: https://github.com/quartiq/rayopt

 - Optical ToolKit, OTK: https://github.com/draustin/otk

"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Optional
import re
from collections.abc import Iterator
import numpy as np

from zmxtools.definitions import (array_like, array_type, asarray, const_c,
                                  FileLike, PathLike, MaterialLibrary, Material, MaterialResistance)
from zmxtools.parser import OrderedCommandDict, Command
from zmxtools import log

__all__ = ["AgfOrderedCommandDict", "AgfMaterialLibrary", "AgfMaterial"]

log = log.getChild(__name__)


class AgfOrderedCommandDict(OrderedCommandDict):
    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 0) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text extracted from an agf file.

        :param file_contents: The text string extracted from an optical file.
        :param spaces_per_indent: The number of spaces per tab to assumed (default: 0)

        :return: The command dictionary.
        """
        def parse_section(lines: Iterator[str],
                          out: Optional[OrderedCommandDict] = None) -> Tuple[OrderedCommandDict, Optional[Command]]:
            """
            Auxiliary recursive function to parse a section of the file's lines.

            :param lines: The iterable with the lines to parse.
            :param out: The optional collection to add to as the result.

            :return: A collection of commands, corresponding to the lines in this section.
            """
            if out is None:
                out = OrderedCommandDict(spaces_per_indent=spaces_per_indent)
            try:
                current_material: Optional[Command] = None
                while (line := next(lines)) is not None:
                    match = re.match(r"(\s*)(\S+)(\s.*)?", line)
                    if match is None:
                        continue  # skip empty line

                    indent_str, command_name, command_argument = match.groups()
                    if command_argument is not None:
                        command_argument = command_argument[1:]
                    next_command = Command(name=command_name, argument=command_argument)

                    if next_command.name == "NM":
                        out.append(next_command)  # Add as a new material
                        current_material = next_command  # append the following as sub-commands
                    else:
                        if current_material is not None:
                            current_material.append(next_command)
                        else:
                            out.append(next_command)  # Add as a child node
            except StopIteration:
                pass
            return out, None

        return parse_section(file_contents.splitlines().__iter__())[0]


class AgfMaterialLibrary(MaterialLibrary):
    @staticmethod
    def from_str(contents: str, spaces_per_indent: int = 0) -> AgfMaterialLibrary:
        """Parses the text extracted from a .agf file into a material library."""
        return AgfMaterialLibrary(AgfOrderedCommandDict.from_str(contents, spaces_per_indent=spaces_per_indent))

    @staticmethod
    def from_file(input_path_or_stream: FileLike | PathLike,
                  spaces_per_indent: int = 0,
                  encoding: str = 'utf-16') -> AgfMaterialLibrary:
        """
        Reads an agf file into an `AgfMaterialLibrary` representation.

        :param input_path_or_stream: The file to read the optical system from, or its file-path.
        :param spaces_per_indent: The optional number of spaces for indenting glass properties.
        :param encoding: The text-encoding to try first.

        :return: A representation of the material library.
        """
        result = AgfMaterialLibrary(AgfOrderedCommandDict.from_file(input_path_or_stream,
                                                                    spaces_per_indent=spaces_per_indent,
                                                                    encoding=encoding))
        if result.name == "":
            result.name = input_path_or_stream.name
        return result

    def __init__(self, commands: OrderedCommandDict):
        self.commands = commands

        description: str = '\n'.join(_.argument for _ in self.commands["CC"]) if "CC" in self.commands else ""
        material_constructors = [SchottAgfMaterial, Sellmeier1AgfMaterial, HerzbergerAgfMaterial,
                                 Sellmeier2AgfMaterial, ConradyAgfMaterial, Sellmeier3AgfMaterial,
                                 HandbookOfOptics1AgfMaterial, HandbookOfOptics2AgfMaterial,
                                 Sellmeier4AgfMaterial, Extended1AgfMaterial, Sellmeier5AgfMaterial,
                                 Extended2AgfMaterial, Formula13AgfMaterial]
        materials = list[AgfMaterial]()
        if "NM" in self.commands:
            log.debug(f'Parsed {len(self.commands["NM"])} materials.')
            for _ in self.commands["NM"]:
                formula_index = int(float(_.words[1])) - 1  # Some glass names are numbers. Some files contain decimal points in the integer index.
                if 0 <= formula_index < len(material_constructors):
                    materials.append(material_constructors[formula_index](_))
                else:
                    raise ValueError(f"Unknown dispersion formula number {formula_index + 1} for material {_.words[0]}: {_} with coefficients {_['CD']}.")
        super().__init__(name="", description=description, materials=materials)

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands[-1])})"
        # return f"{self.__class__.__name__}(commands={repr(self.commands)})"


def as_float(number_str: str) -> float:
    number_str = number_str.strip()
    if number_str != '-':
        return float(number_str)
    return np.nan

def to_length(numbers: Sequence[float], length: int, pad_value: float = np.nan):
    """Auxiliary function to ensure that we always have the correct number of values."""
    return asarray([*numbers, *([pad_value] * (length - len(numbers)))][:length])


class AgfMaterial(Material):
    """A class to represent a material in a Zemax AGF file."""
    description: str
    glass_code: str
    production_frequency: int  # melt frequency
    ignore_thermal_expansion: bool

    _permittivity_coefficients_um: List[float]

    __thermal_coefficients: List[float]
    __wavenumber_transmission_thickness: array_like
    __transmission: array_like
    __transmission_thickness: array_like

    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        """
        Construct a new glass material based from a command and its child-nodes that represent the corresponding lines
        in the file.

        :param command: The command that describes this glass.
        :param temperature: The temperature of this material in degrees Kelvin, K.
        :param pressure: The pressure of this material in Pa = N/m^2.
        """
        self.command = command

        name = self.command.words[0]
        formula_index, glass_code, self.__refractive_index_d, self.__constringence, exclude_sub, status_index, self.production_frequency = \
            to_length([as_float(_) for _ in self.command.words[1:]], 7, pad_value=0.0)
        # formula_index = int(formula_index)
        # self.glass_code = self.command.words[1]
        status_index = int(status_index)
        statuses = ["standard", "preferred", "special", "obsolete", "melt"]
        # self.status = f"{statuses[status_index - 1]} ({status_index})"

        # Extract the child-node arguments now
        self.description = self.command["GC", 0].argument if "GC" in self.command else ""
        alpham3070, alpha20300, density_g_cm3, delta_relative_partial_dispersion_g_F, ignore_thermal_expansion = to_length(
            self.command["ED", 0].numbers if "ED" in self.command else list[float](), 5, pad_value=0.0)
        self.density = density_g_cm3 * 1e3
        # self.ignore_thermal_expansion = ignore_thermal_expansion != 0

        # The permittivity formula coefficients for wavelengths in meters
        self._permittivity_coefficients_um = self.command["CD", 0].numbers if "CD" in self.command else list[float]()

        self.__thermal_coefficients = to_length(self.command["TD", 0].numbers if "TD" in self.command else list[float](), 6, pad_value=0.0).real
        relative_cost, climatic_resistance_index, stain_resistance_index, acid_resistance_index, \
            alkali_resistance_index, phosphate_resistance_index \
            = to_length([int(_) for _ in self.command["OD", 0].numbers] if "OD" in self.command else list[int](),
                        6, pad_value=-1)  # Cost and chemical info

        self.resistance = MaterialResistance(climate=climatic_resistance_index, stain=stain_resistance_index,
                                             acid=acid_resistance_index, alkali=alkali_resistance_index,
                                             phosphate=phosphate_resistance_index)

        wavelength_limits = np.sort(to_length(self.command["LD", 0].numbers, 2, pad_value=0))[::-1] * 1e-6 \
            if "CD" in self.command else asarray([np.inf, 0.0])  # Convert micrometers to meters, in reverse order
        wavenumber_limits = 2 * np.pi / wavelength_limits

        wavelength_transmission_thickness = sorted([to_length(_.numbers, 3, pad_value=1.0) for _ in self.command["IT"]], key=lambda _: _[0].real)
        specified_wavenumbers = [2 * np.pi / (_[0].real * 1e-6) for _ in wavelength_transmission_thickness[::-1]]
        specified_transmissions = [_[1].real for _ in wavelength_transmission_thickness[::-1]]
        specified_thicknesses = [_[2].real * 1e-3 for _ in wavelength_transmission_thickness[::-1]]  # mm to meters
        if len(specified_wavenumbers) > 0:
            self.__transmission = lambda _: np.interp(_.real, specified_wavenumbers, specified_transmissions)
            self.__transmission_thickness = lambda _: np.interp(_.real, specified_wavenumbers, specified_thicknesses)
        else:  # Model as perfect transmission
            self.__transmission = lambda _: _.real * 0.0 + 1.0
            self.__transmission_thickness = lambda _: _.real * 0.0 + 1.0

        super().__init__(name=name, wavenumber_limits=wavenumber_limits, temperature=temperature, pressure=pressure)

    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        """Either this or the _real_wavelength_um method must be overridden in subclass."""
        return self._real_reference_refractive_index_um(w)**2

    def _real_reference_refractive_index_um(self, w: array_type) -> array_type:
        """Either this or the _real_permittivity_um method must be overridden in subclass."""
        return self._real_reference_permittivity_um(w)**0.5

    def refractive_index(self,
                         wavenumber: Optional[array_like] = None,
                         wavelength: Optional[array_like] = None,
                         angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the real refractive index for this material at the specified wavenumbers, wavelengths,
        and angular_frequencies. Only one must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The refractive index, adjusted for this material's temperature and pressure.
        """
        if wavelength is None:
            if wavenumber is None:
                wavenumber = asarray(angular_frequency) / const_c
            else:
                wavenumber = asarray(wavenumber)
            wavelength = 2 * np.pi / wavenumber
        else:
            wavelength = asarray(wavelength)
        # Get n and ε at reference temperature
        real_n = self._real_reference_refractive_index_um(wavelength / 1e-6).real
        real_permittivity = real_n**2

        # Correct for temperature difference with the reference temperature for this material
        thc = [*self.__thermal_coefficients[:3],
               self.__thermal_coefficients[3] * 1e-6**2,
               self.__thermal_coefficients[4] * 1e-6**2,
               self.__thermal_coefficients[5] * 1e-6]  # in meters
        reference_wavelength = thc[5]
        reference_temperature = 20.0 + 273.15
        d_temp = self.temperature.real - reference_temperature

        thermal_correction = thc[0] * d_temp + thc[1] * d_temp**2 + thc[2] * d_temp**3 \
                             + ((thc[3] * d_temp + thc[4] * d_temp**2) / (wavelength.real**2 - reference_wavelength**2))
        thermal_correction *= 0.5 * (real_permittivity - 1) / real_permittivity

        return real_n * (1 + thermal_correction)

    def extinction_coefficient(self,
                                wavenumber: Optional[array_like] = None,
                                wavelength: Optional[array_like] = None,
                                angular_frequency: Optional[array_like] = None) -> array_type:
        """
        Calculates the extinction (or absorption) coefficient, κ, for this material at the specified
        wavenumbers, wavelengths, and angular_frequencies. Only one must be specified.

        :param wavenumber: The wavenumbers in rad/m.
        :param wavelength: The wavelength in units of m.
        :param angular_frequency: The angular velocity in units of rad/s.

        :return: The extinction coefficient κ.
        """
        if wavenumber is None:
            if wavelength is None:
                wavelength = const_c * 2 * np.pi / asarray(angular_frequency)
            else:
                wavelength = asarray(wavelength)
            wavenumber = 2 * np.pi / wavelength
        else:
            wavenumber = asarray(wavenumber)
        wavenumber = wavenumber.real

        return -0.5 / wavenumber * np.log(self.__transmission(wavenumber)) / self.__transmission_thickness(wavenumber)

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.command)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.command)})"


class SchottAgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 6, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] * w2) \
            + (c[2] * w2**-1) \
            + (c[3] * w2**-2) \
            + (c[4] * w2**-3) \
            + (c[5] * w2**-4)


class Sellmeier1AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 6, pad_value=0.0)
        w2 = w**2
        return 1.0 \
            + (c[0] * w2 / (w2 - c[1])) \
            + (c[2] * w2 / (w2 - c[3])) \
            + (c[4] * w2 / (w2 - c[5]))


class Sellmeier2AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 5, pad_value=0.0)
        w2 = w**2
        return 1.0 \
            + c[0] \
            + (c[1] * w2 / (w2 - (c[2])**2)) \
            + (c[3] / (w2 - (c[4])**2))


class Sellmeier3AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 8, pad_value=0.0)
        w2 = w**2
        return 1.0 \
            + (c[0] * w2 / (w2 - c[1])) \
            + (c[2] * w2 / (w2 - c[3])) \
            + (c[4] * w2 / (w2 - c[5])) \
            + (c[6] * w2 / (w2 - c[7]))


class Sellmeier4AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 5, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] * w2 / (w2 - c[2])) \
            + (c[3] * w2 / (w2 - c[4]))


class Sellmeier5AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 10, pad_value=0.0)
        w2 = w**2
        return 1.0 \
            + (c[0] * w2 / (w2 - c[1])) \
            + (c[2] * w2 / (w2 - c[3])) \
            + (c[4] * w2 / (w2 - c[5])) \
            + (c[6] * w2 / (w2 - c[7])) \
            + (c[8] * w2 / (w2 - c[9]))


class HerzbergerAgfMaterial(AgfMaterial):
    def _real_reference_refractive_index_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 6, pad_value=0.0)
        w2 = w**2
        lw = 1.0 / (w2 - 0.028)
        return c[0] \
            + (c[1] * lw) \
            + (c[2] * lw**2) \
            + (c[3] * w2) \
            + (c[4] * w2**2) \
            + (c[5] * w2**3)


class ConradyAgfMaterial(AgfMaterial):
    def _real_reference_refractive_index_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 3, pad_value=0.0)
        return c[0] + (c[1] / w) + (c[2] / w**3.5)


class HandbookOfOptics1AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 4, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] / (w2 - c[2])) \
            - (c[3] * w2)


class HandbookOfOptics2AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 4, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] * w2 / (w2 - c[2])) \
            - (c[3] * w2)


class Extended1AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 8, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] * w2) \
            + (c[2] * w2**-1) \
            + (c[3] * w2**-2) \
            + (c[4] * w2**-3) \
            + (c[5] * w2**-4) \
            + (c[6] * w2**-5) \
            + (c[7] * w2**-6)


class Extended2AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 8, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] * w2) \
            + (c[2] * w2**-1) \
            + (c[3] * w2**-2) \
            + (c[4] * w2**-3) \
            + (c[5] * w2**-4) \
            + (c[6] * w2**2) \
            + (c[7] * w2**3)


class Formula13AgfMaterial(AgfMaterial):
    def _real_reference_permittivity_um(self, w: array_type) -> array_type:
        c = to_length(self._permittivity_coefficients_um, 9, pad_value=0.0)
        w2 = w**2
        return c[0] \
            + (c[1] * w2) \
            + (c[2] * w2**2) \
            + (c[3] * w2**-1) \
            + (c[4] * w2**-2) \
            + (c[5] * w2**-3) \
            + (c[6] * w2**-4) \
            + (c[7] * w2**-5) \
            + (c[8] * w2**-6)
