from __future__ import annotations

from collections.abc import Iterator
import re
import math
from typing import Union, Tuple, Optional

from zmxtools.definitions import (FileLike, PathLike, Material, Vacuum, MaterialLibrary, Surface, OpticalDesign)
from zmxtools.parser import OrderedCommandDict, Command
from zmxtools import log

log = log.getChild(__name__)

__all__ = ["ZmxOrderedCommandDict", "ZmxOpticalDesign", "ZmxSurface"]


class ZmxOrderedCommandDict(OrderedCommandDict):
    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 2) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text extracted from a zmx file.

        :param file_contents: The text string extracted from an optical file.
        :param spaces_per_indent: The number of spaces per indent to assumed

        :return: The command dictionary.
        """
        def parse_section(lines: Iterator[str],
                          parent_indent: int = -1, section_indent: int = 0,
                          out: Optional[OrderedCommandDict] = None) -> Tuple[OrderedCommandDict, Optional[Command]]:
            """
            Auxiliary recursive function to parse a section of the file's lines.

            :param lines: The iterable with the lines to parse.
            :param parent_indent: The indent of the enclosing section.
            :param section_indent: The indent of the current section.
            :param out: The optional collection to add to as the result.

            :return: A collection of commands, corresponding to the lines in this section.
            """
            if out is None:
                out = OrderedCommandDict(spaces_per_indent=spaces_per_indent)
            try:
                while (line := next(lines)) is not None:
                    match = re.match(r"(\s*)(\S+)(\s.*)?", line)
                    if match is None:
                        continue  # skip empty line
                    indent_str, command_name, command_argument = match.groups()
                    if command_argument is not None:
                        command_argument = command_argument[1:]
                    if '\t' not in indent_str:
                        indent = len(indent_str)
                    else:  # Replace tabs with spaces before counting indentation
                        indent = sum((spaces_per_indent - (_ % spaces_per_indent)) if c == '\t' else 1 for _, c in enumerate(indent_str))
                    next_command = Command(name=command_name, argument=command_argument)
                    if indent <= parent_indent:
                        return out, next_command  # pass next command to
                    elif parent_indent < indent <= section_indent:  # be very forgiving
                        out.append(next_command)
                    else:  # indent > section_indent:  recurse
                        out[-1].children, next_command = parse_section(
                            lines,
                            parent_indent=section_indent, section_indent=indent,
                            out=OrderedCommandDict([next_command], spaces_per_indent=spaces_per_indent))
                        if next_command is not None:
                            out.append(next_command)
            except StopIteration:
                pass
            return out, None

        return parse_section(file_contents.splitlines().__iter__())[0]


class ZmxOpticalDesign(OpticalDesign):
    @staticmethod
    def from_str(contents: str, spaces_per_indent: int = 2) -> ZmxOpticalDesign:
        """Parses the text extracted from a .zmx file into an `OpticalDesign`."""
        return ZmxOpticalDesign(ZmxOrderedCommandDict.from_str(contents, spaces_per_indent=spaces_per_indent))

    @staticmethod
    def from_file(input_path_or_stream: Union[FileLike, PathLike],
                  spaces_per_indent: int = 2,
                  encoding: str = 'utf-16') -> ZmxOpticalDesign:
        """
        Reads a zmx file into an `OpticalDesign` representation.

        :param input_path_or_stream: The file to read the optical system from, or its file-path.
        :param spaces_per_indent: The optional number of spaces per indent/tab.
        :param encoding: The text-encoding to try first.

        :return: A representation of the optical system.
        """
        return ZmxOpticalDesign(ZmxOrderedCommandDict.from_file(input_path_or_stream,
                                                                spaces_per_indent=spaces_per_indent,
                                                                encoding=encoding))

    def __init__(self, commands: OrderedCommandDict):
        self.commands = commands

        self.version = self.commands["VERS", 0].argument if "VERS" in self.commands else ""
        self.sequential = True
        if "MODE" in self.commands:
            mode = self.commands["MODE", 0].words[0]
            if mode != "SEQ":
                if mode == "NSC":
                    log.warning(f"Non-sequential mode not implemented.")
                else:
                    log.warning(f"Unrecognized mode {mode}.")
                self.sequential = False
        self.name = self.commands["NAME", 0].argument if "NAME" in self.commands else ""
        self.author = self.commands["AUTH", 0].argument if "AUTH" in self.commands else ""
        self.description = '\n'.join(_.argument.replace('\n', '') for _ in self.commands.sort_and_merge("NOTE")) \
            if "NOTE" in self.commands else ""
        self.field_comment = self.commands["FCOM", 0].argument if "FCOM" in self.commands else ""
        self.unit: float = 1.0
        if "UNIT" in self.commands:
            unit_code = self.commands["UNIT", 0].argument.split(maxsplit=1)[0]
            unit_dict = {"UM": 1e-6, "MM": 1e-3, "CM": 1e-2, "DM": 100e-3, "METER": 1.0, "ME": 1.0, "M": 1.0,
                         "DA": 10.0, "HM": 100.0, "KM": 1e3, "GM": 1e9, "TM": 1e12,
                         "IN": 25.4e-3, "FEET": 304.8e-3, "FT": 304.8e-3, "FE": 304.8e-3, "FO": 304.8e-3}
            if unit_code in unit_dict:
                self.unit = unit_dict[unit_code]
            else:
                log.warning(f"Unrecognized unit code {unit_code}. Defaulting to 1m.")
        self.surfaces = [ZmxSurface(s.children) for s in self.commands.sort_and_merge("SURF")]
        self.wavelengths = self.commands["WAVL", 0].numbers if "WAVL" in self.commands else list[float]()  # "WAVM" doesn't seem very reliable. Perhaps this depends on the version?
        self.wavelength_weights = self.commands["WWGT", 0].numbers if "WWGT" in self.commands else list[float]()
        if len(self.wavelength_weights) < len(self.wavelengths):
            self.wavelength_weights = [*self.wavelength_weights, *([1.0] * (len(self.wavelengths) - len(self.wavelength_weights)))]
        if len(self.wavelengths) == 0 and "WAVM" in self.commands:  # This seems to be the new way, but it contains many unused wavelengths as well
            wavelengths_and_weights = [_.numbers[:2] for _ in self.commands.sort_and_merge("WAVM")]
            unique_wavelengths = set(_[0] for _ in wavelengths_and_weights)
            nb_occurences = [sum(u == _[0] for _ in wavelengths_and_weights) for u in unique_wavelengths]
            unique_wavelengths = [_ for _, n in zip(unique_wavelengths, nb_occurences) if n == 1]
            wavelengths_and_weights = [_ for _ in wavelengths_and_weights if _[0] in unique_wavelengths]

            self.wavelengths = [_[0] for _ in wavelengths_and_weights]
            self.wavelength_weights = [_[1] for _ in wavelengths_and_weights]
        self.material_libraries = list[MaterialLibrary]()
        if "GCAT" in self.commands:
            for name in self.commands["GCAT", 0].words:
                self.material_libraries.append(MaterialLibrary(name=name, description=name))
        self.coating_filenames = list[str]()
        if "COFN" in self.commands:
            file_names = self.commands["COFN", 0].words
            if file_names[0] == "QF":
                file_names = file_names[1:]
            self.coating_filenames += file_names
        self.numerical_aperture = 1.0
        if "FNUM" in self.commands:
            f_number = self.commands["FNUM", 0].numbers[0]
            self.numerical_aperture_image = 2.0 / f_number  # todo: account for refractive index of object or image space?
        if "OBNA" in self.commands:
            self.numerical_aperture_object = self.commands["OBNA", 0].numbers[0]
        if "ENPD" in self.commands:
            pupil_radius_object = self.commands["ENPD", 0].numbers[0] / 2.0
        if "EFFL" in self.commands:
            effective_focal_length = self.commands["EFFL", 0].numbers[0]
        if "FTYP" in self.commands:
            field_type = self.commands["FTYP", 0].numbers[0]  # []
            field_as_height = (field_type % 2) == 1   # angle: False, height: True
            field_at_image = (field_type // 2) == 1  # object: False, image: True
        # field also uses VDXN, VDYN, VCXN, VXYN, VANN, VWGN, VWGT

        # Make all units meters
        self.wavelengths = [_ * 1e-6 for _ in self.wavelengths]
        for s in self.surfaces:
            s.curvature /= self.unit
            s.distance *= self.unit
            s.radius *= self.unit

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"


class ZmxSurface(Surface):
    """A class to represent a thin surface between two volumes as read from a .zmx file."""
    def __init__(self, commands: OrderedCommandDict):
        """
        Construct a new surface based from a command dictionary that represents the corresponding lines in the file.

        :param commands: The command dictionary.
        """
        self.commands = commands

        self.material = Vacuum()
        self.type = self.commands["TYPE", 0].argument if "TYPE" in self.commands else "STANDARD"
        # Types: "STANDARD" , "EVENASPH", "TOROIDAL", "XOSPHERE", "COORDBRK", "TILTSURF", "PARAXIAL", "DGRATING"
        self.curvature = self.commands["CURV", 0].numbers[0] if "CURV" in self.commands else 0.0
        self.coating = self.commands["COAT", 0].words[0] if "COAT" in self.commands and len(self.commands["COAT", 0].words) > 0 else ""
        self.radius = self.commands["DIAM", 0].numbers[0] / 2.0 if "DIAM" in self.commands else math.inf
        self.stop = "STOP" in self.commands
        self.distance = self.commands["DISZ", 0].numbers[0] if "DISZ" in self.commands else math.inf
        self.comment = self.commands["COMM", 0].argument if "COMM" in self.commands else ""
        glass_name = self.commands["GLAS", 0].words[0] if "GLAS" in self.commands and len(self.commands["GLAS", 0].words) else ""
        self.material = Material(name=glass_name)
        self.reflect = glass_name == "MIRROR"  # Not "MIRR" command for some reason
        # self.floating_aperture = self.commands["FLAP", 0].numbers if "FLAP" in self.commands else 0.0
        self.conic_constant = self.commands["CONI", 0].numbers if "CONI" in self.commands else 0.0
        toroidal_x = self.commands["XDAT", 0].numbers if "XDAT" in self.commands else []
        toroidal_y = self.commands["YDAT", 0].numbers if "YDAT" in self.commands else []
        aperture_offsets = self.commands["OBDC", 0].numbers[:2] if "OBDC" in self.commands else []

        # STANDARD surface: z**2 == 2 * r * self.curvature - (1 + self.conic_constant) * r**2
        sag = lambda r: self.curvature * r**2 / (1 + (1 - (1 + self.conic_constant) * (self.curvature * r)**2)**0.5)

    @property
    def eccentricity(self):
        return (-self.conic_constant)**0.5

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"
