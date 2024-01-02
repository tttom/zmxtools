"""
Parse Zemax Glass AGF files.

The following projects have been useful references:

 - RefractiveIndex.info, https://github.com/polyanskiy/refractiveindex.info-database

 - The ZemaxGlass project, https://github.com/nzhagen/zemaxglass

 - RayOpt: https://github.com/quartiq/rayopt

 - Optical ToolKit, OTK: https://github.com/draustin/otk

"""
from __future__ import annotations

from typing import Tuple, Optional, Callable
import re
from collections.abc import Iterator
import numpy as np

from zmxtools.definitions import array_like, array_type, asarray, FileLike, PathLike
from zmxtools.optical_design.material import MaterialLibrary, Material, FunctionMaterial, PolynomialMaterial, MaterialResistance
from zmxtools.parser import OrderedCommandDict, Command
from zmxtools.utils import to_length
from zmxtools import log

__all__ = ["AgfOrderedCommandDict", "AgfMaterialLibrary", "AgfMixin"]

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
                                 Extended2AgfMaterial, Extended3AgfMaterial]
        materials = list[Material]()
        if "NM" in self.commands:
            log.debug(f'Parsed {len(self.commands["NM"])} materials.')
            for command in self.commands["NM"]:
                formula_index = int(float(command.words[1])) - 1  # Some glass names are numbers. Some files contain decimal points in the integer index.
                if 0 <= formula_index < len(material_constructors):
                    materials.append(material_constructors[formula_index](command))
                else:
                    raise ValueError(f"Unknown dispersion formula number {formula_index + 1}" +
                                     f" for material {command.words[0]}: {command} with coefficients {command['CD']}.")
        super().__init__(name="", description=description, materials=materials)

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"


def as_float(number_str: str) -> float:
    number_str = number_str.strip()
    if number_str != '-':
        return float(number_str)
    return np.nan


class AgfMixin:
    """A mix-in class to represent a material in a Zemax AGF file."""
    command: Command

    name: str
    wavenumber_limits: array_type
    description: str
    density: float  # in kg / m^3
    glass_code: str
    production_frequency: int  # melt frequency
    ignore_thermal_expansion: bool
    resistance: MaterialResistance

    permittivity_coefficients: array_type
    specified_wavenumber_transmission_thickness: array_type
    thermal_coefficients: array_type

    def __init__(self, command: Command):
        """
        Construct a new glass material based from a command and its child-nodes that represent the corresponding lines
        in the file. Provides the method adjust_refractive_index_function amongst other properties.

        :param command: The command that describes this glass.
        """
        self.command = command

        self.name = self.command.words[0]
        formula_index, glass_code, self.__refractive_index_d, self.__constringence, exclude_sub, status_index, self.production_frequency = \
            to_length([as_float(_) for _ in self.command.words[1:]], 7, value=0.0)
        # self.formula_index: int = int(formula_index)
        self.glass_code = self.command.words[1]
        # status_index = int(status_index)
        # statuses = ["standard", "preferred", "special", "obsolete", "melt"]
        # self.status = f"{statuses[status_index - 1]} ({status_index})"

        # Extract the child-node arguments now
        self.description = self.command["GC", 0].argument if "GC" in self.command else ""
        alpham3070, alpha20300, density_g_cm3, delta_relative_partial_dispersion_g_F, ignore_thermal_expansion = \
            to_length(self.command["ED", 0].numbers if "ED" in self.command else list[float](), 5, value=0.0)
        self.density = density_g_cm3 * 1e3
        # self.ignore_thermal_expansion = ignore_thermal_expansion != 0

        relative_cost, climatic_resistance_index, stain_resistance_index, acid_resistance_index, \
            alkali_resistance_index, phosphate_resistance_index \
            = to_length([int(_) for _ in self.command["OD", 0].numbers] if "OD" in self.command else list[int](),
                        6, value=-1)  # Cost and chemical info
        self.resistance = MaterialResistance(climate=climatic_resistance_index, stain=stain_resistance_index,
                                             acid=acid_resistance_index, alkali=alkali_resistance_index,
                                             phosphate=phosphate_resistance_index)

        # The permittivity formula coefficients for wavelengths in micrometers
        self.permittivity_coefficients = asarray(self.command["CD", 0].numbers
                                                 if "CD" in self.command else list[float]()).real

        # Process extinction information
        specified_wavelength_transmission_thickness = sorted(
            [to_length(_.numbers, 3, value=1.0).real for _ in command["IT"]], key=lambda _: _[0].real)
        self.specified_wavenumber_transmission_thickness = asarray([
            (2 * np.pi / (w * 1e-6), tr, th * 1e-3)  # convert to m
            for w, tr, th in specified_wavelength_transmission_thickness[::-1]
        ]).real

        # Determine thermal correction factor for refractive index
        self.thermal_coefficients = to_length(
            self.command["TD", 0].numbers if "TD" in self.command else list[float](), 6, value=0.0).real

        # Extract the refractive index model validity limits
        wavelength_limits = np.sort(to_length(self.command["LD", 0].numbers, 2, value=0))[::-1] * 1e-6 \
            if "CD" in self.command else asarray([np.inf, 0.0])  # Convert micrometers to meters, in reverse order
        self.wavenumber_limits = 2 * np.pi / wavelength_limits

    def adjust_refractive_index_function(self, refractive_index_at_reference_function: Callable[[array_type], array_type],
                                         wavenumber: array_type, temperature: array_type, pressure: array_type) -> array_type:
        """
        An auxiliary method that returns the complex refractive index as a function of the reference refractive index
        function, the wavenumber, temperature, and pressure.

        The method uses self.thermal_coefficients (in um), self.specified_wavenumber_transmission_thickness (in m)

        :param refractive_index_at_reference_function: The refractive index at the reference temperature and pressure conditions.
        :param wavenumber: The wavenumber to calculate .
        :param temperature: The temperature to compute the refractive index at.
        :param pressure: The pressure to compute the refractive index at.

        :return: The corrected complex refractive index.
        """
        # Calculate the refractive index at the reference temperature and pressure
        refractive_index_at_reference = refractive_index_at_reference_function(wavenumber)

        # Calculate the thermal correction factor from the wavenumber, self.thermal_coefficients
        coeff_t = self.thermal_coefficients[:3]
        coeff_tw_um2 = self.thermal_coefficients[3:5]  # in :math:`\mu m^2`
        reference_wavelength_um = self.thermal_coefficients[5]  # in um
        reference_temperature = 20.0 + 273.15
        d_temp = temperature.real - reference_temperature

        wavelength_um = 2 * np.pi / wavenumber.real / 1e-6
        thermal_correction = coeff_t[0] * d_temp + coeff_t[1] * d_temp**2 + coeff_t[2] * d_temp**3 \
                             + ((coeff_tw_um2[0] * d_temp + coeff_tw_um2[1] * d_temp**2)
                                / (wavelength_um**2 - reference_wavelength_um**2))
        refractive_index_at_reference_relative = refractive_index_at_reference  # TODO?: / refractive_index_air_at_reference  # from https://support.zemax.com/hc/en-us/articles/1500005576002-How-OpticStudio-calculates-refractive-index-at-arbitrary-temperatures-and-pressures
        thermal_correction = thermal_correction * 0.5 * (refractive_index_at_reference_relative ** 2 - 1) / refractive_index_at_reference_relative

        # Calculate the extinction coefficient at a given wavenumber
        if self.specified_wavenumber_transmission_thickness.shape[0] > 0:  # by interpolationv
            specified_wavenumbers, specified_transmissions, specified_thicknesses = \
                self.specified_wavenumber_transmission_thickness.transpose()
            transmission = lambda _: np.interp(_.real, specified_wavenumbers, specified_transmissions)
            transmission_thickness = lambda _: np.interp(_.real, specified_wavenumbers, specified_thicknesses)
        else:  # model as perfect transmission
            transmission = lambda _: _.real * 0.0 + 1.0
            transmission_thickness = lambda _: _.real * 0.0 + 1.0
        # Determine the extinction coefficient from the wavenumber, transmission, and thickness
        extinction_coefficient = lambda wavenumber: \
            -0.5 / wavenumber * np.log(transmission(wavenumber)) / transmission_thickness(wavenumber)

        return refractive_index_at_reference + thermal_correction + 1j * extinction_coefficient(wavenumber)

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.command)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.command)})"


class SchottAgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 1 with relative permittivity

        .. math::
            \epsilon_r(\lambda) := c_0 + c_1 \lambda^2 + \sum_{i=2} c_i \lambda^{2(i-1)},

        where :math:`c_i` are the coefficient factors and :math:`\lambda^2` is the squared wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                         temperature=temperature, pressure=pressure,
                                    factors_um=self.permittivity_coefficients, exponents=[0, 2, -2, -4, -6, -8],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Sellmeier1AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 2 with relative permittivity

        .. math::
            \epsilon_r(\lambda) := 1 + \sum_{i=1} c_i \frac{\lambda^2}{\lambda^2 - p_i},

        where :math:`c_i` are the coefficient factors and :math:`p_i` are the poles, while `\lambda^2` is the squared
        wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=[1.0, *self.permittivity_coefficients[::2]],
                                    exponents=[0, *([2] * (1 + len(self.permittivity_coefficients) // 2))],
                                    poles_um2=[np.nan, *self.permittivity_coefficients[1::2]],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Sellmeier2AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        TODO: Find a test case for this type of glass.
        Construct a material using AGF formula 4 with relative permittivity

        .. math::
            \epsilon_r(\lambda) := 1 + c_1 + c_2\frac{\lambda^2}{\lambda^2 - p_2} + \sum_{i=3} c_i \frac{1}{\lambda^2 - p_i},

        where :math:`c_i` are the coefficient factors and :math:`p_i` are the poles, while `\lambda^2` is the squared
        wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=[1.0, self.permittivity_coefficients[0], *self.permittivity_coefficients[1::2]],
                                    exponents=[0, 0, 2, 0],
                                    poles_um2=[np.nan, np.nan, *(self.permittivity_coefficients[2::2] ** 2)],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Sellmeier3AgfMaterial(Sellmeier1AgfMaterial):
    r"""
    Represents a material using AGF formula 6 with relative permittivity

    .. math::
            \epsilon_r(\lambda) := 1 + \sum_{i=1} c_i \frac{\lambda^2}{\lambda^2 - p_i},

    where :math:`c_i` are the coefficient factors and :math:`p_i` are the poles, while `\lambda^2` is the squared
    wavelength in micrometers.

    """


class Sellmeier4AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 9 with relative permittivity

        .. math::
            \epsilon_r(\lambda) := c_0 + \sum_{i=1} c_i \frac{\lambda^2}{\lambda^2 - p_i},

        where :math:`c_i` are the coefficient factors and :math:`p_i` are the poles of $\lambda^2$, where $\lambda$ is
        the wavelength in :math:`\mu m`.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=[self.permittivity_coefficients[0], *self.permittivity_coefficients[1::2]],
                                    exponents=[0, *([2] * (len(self.permittivity_coefficients) // 2))],
                                    poles_um2=[np.nan, *self.permittivity_coefficients[2::2]],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Sellmeier5AgfMaterial(Sellmeier1AgfMaterial):
    """
    Represents a material using AGF formula 11 with relative permittivity
    """


class HerzbergerAgfMaterial(AgfMixin, FunctionMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None, pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 3, for a **reference refractive index** that is given by:

        .. math::
            l(w) = w^2 - 0.028
            n(\lambda) := c_0 + \frac{c_1}{l(w)} + \frac{c_2}{l(w)^2} + c_3 w^2 + c_4 w^4 + c_5 w^6,

        where :math:`c_i` are the coefficient factors and :math:`l(\lambda)` is the shifted-squared wavelength in
        :math:`mu m^2`, while :math:`\lambda^2` is the squared wavelength in micrometers.

        :param name: The name of this material.
        :param temperature: The temperature this material is at in degrees K.
        :param pressure: The pressure this material is at in Pascals [Pa].
        :param factors_um: The coefficient factors for a polynomial in the micrometer-wavelengths
        """
        AgfMixin.__init__(self, command)

        def refractive_index_at_reference_function(wavenumber: array_type) -> array_type:
            """
            The refractive index at the reference temperature and pressure, as a function of wavenumber.
            """
            wavelength_um = 2 * np.pi / wavenumber / 1e-6
            wavelength_um2 = wavelength_um ** 2
            l_um2 = 1.0 / (wavelength_um2 - 0.028)

            l_terms = sum((factor_um * (l_um2 ** exponent)
                           for factor_um, exponent in zip(self.permittivity_coefficients[:3], range(3))),
                          start=0.0)
            w_terms = sum((factor_um * (wavelength_um2 ** exponent)
                           for factor_um, exponent in zip(self.permittivity_coefficients[3:], range(1, 4))),
                          start=0.0)
            return l_terms + w_terms

        FunctionMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                  temperature=temperature, pressure=pressure,
                                  complex_refractive_index_function=
                                  lambda k, t, p: self.adjust_refractive_index_function(
                                      refractive_index_at_reference_function, k, t, p))


class ConradyAgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 5 with reference refractive index

        .. math:
            n(\lambda) := c_0 + \frac{c_1}{\lambda} + \frac{c_2}{\lambda^{7/2}},

        where :math:`c_i` are the coefficient factors and :math:`\lambda` is the wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    refractive_index_model=True,
                                    factors_um=self.permittivity_coefficients, exponents=[0, -1, -3.5],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class HandbookOfOptics1AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 7 with relative permittivity

        .. math:
            \epsilon_r(\lambda) := c_0 + c_1\frac{1}{\lambda^2 - p_i} - c_2 \lambda^2,

        where :math:`c_i` are the coefficient factors, :math:`p_i` are the poles in :math:`\lambda^2`, and `\lambda` is
        wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=[*self.permittivity_coefficients[:2], -self.permittivity_coefficients[3]],
                                    exponents=[0, 0, 2],
                                    poles_um2=[np.nan, self.permittivity_coefficients[2], np.nan],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class HandbookOfOptics2AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material using AGF formula 8 with relative permittivity

        .. math:
            \epsilon_r(\lambda) := c_0 + c_1\frac{\lambda^2}{\lambda^2 - p_i} - c_2 \lambda^2,

        where :math:`c_i` are the coefficient factors, :math:`p_i` are the poles in :math:`\lambda^2`, and `\lambda` is
        wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=[*self.permittivity_coefficients[:2], -self.permittivity_coefficients[3]],
                                    exponents=[0, 2, 2],
                                    poles_um2=[np.nan, self.permittivity_coefficients[2], np.nan],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Extended1AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        TODO: Find a test case for this material
        Construct a material with formula 10 with relative permittivity

        .. math:
            \epsilon_r(\lambda) := c_0 + c_1\lambda^2 + \sum_{i=2}\frac{c_i}{\lambda^{2(i-1)}},

        where :math:`c_i` are the coefficient factors, :math:`p_i` are the poles in :math:`\lambda^2`, and `\lambda` is
        wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=self.permittivity_coefficients,
                                    exponents=[0, 2, -2, -4, -6, -8, -10, -12],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Extended2AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material with formula 12 with relative permittivity

        .. math:
            \epsilon_r(\lambda) = c_0 + c_1\lambda^2 + \sum_{i=2}^{5}\frac{c_i}{\lambda^{2(i-1)}} + \sum_{i=6}c_i\lambda^{2(i-4)},

        where :math:`c_i` are the coefficient factors and `\lambda` is the wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=self.permittivity_coefficients,
                                    exponents=[0, 2, -2, -4, -6, -8, 4, 6],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)


class Extended3AgfMaterial(AgfMixin, PolynomialMaterial):
    def __init__(self, command: Command,
                 temperature: Optional[array_like] = None,
                 pressure: Optional[array_like] = None):
        r"""
        Construct a material with formula 13 with relative permittivity

        .. math:
            \epsilon_r(\lambda) := c_0 + c_1\lambda^2 + c_2\lambda^4 + \sum_{i=3}\frac{c_i}{\lambda^{2(i-2)}},

        where :math:`c_i` are the coefficient factors and `\lambda` is the wavelength in micrometers.

        :param command: The command dictionary to extract name and limits from.
        :param temperature: The temperature of this material.
        :param pressure: The pressure on this material.
        """
        AgfMixin.__init__(self, command)
        PolynomialMaterial.__init__(self, name=self.name, wavenumber_limits=self.wavenumber_limits,
                                    temperature=temperature, pressure=pressure,
                                    factors_um=self.permittivity_coefficients,
                                    exponents=[0, 2, 4, -2, -4, -6, -8, -10, -12],
                                    adjust_refractive_index_function=self.adjust_refractive_index_function)
