from __future__ import annotations

from typing import Union, Sequence, Tuple, Optional, Self
from collections.abc import Iterator
from collections import defaultdict
import re
import math

from zmxtools.definitions import FileLike, PathLike, Material, Vacuum, MaterialLibrary, Surface, OpticalDesign
from zmxtools import log

log = log.getChild(__name__)

__all__ = ["ZmxOpticalDesign"]


class OrderedCommandDict:
    def __init__(self, commands: Sequence[Command] = tuple(), spaces_per_tab: int = 2):
        """
        Construct a new command dictionary from a sequence of commands.

        :param commands: A sequence of commands, which will be kept in order.
        :param spaces_per_tab: The number of tabs to use when converting to str with __str__()
        """
        assert all(_ is not None for _ in commands), f"{self.__class__.__name__} expected as sequence of Commands, not {commands}."
        self.__commands = list(commands)
        self.spaces_per_tab: int = spaces_per_tab  # For __str__()

    @staticmethod
    def from_str(file_contents: str, spaces_per_tab: int = 2) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text.

        :param file_contents: The text string extracted from an optical file.
        :param spaces_per_tab: The number of spaces per tab to assumed

        :return: The command dictionary.
        """
        def parse_section(lines: Iterator[str],
                          parent_indent: int = -1, section_indent: int = 0, spaces_per_tab: int = 2,
                          out: Optional[OrderedCommandDict] = None) -> Tuple[OrderedCommandDict, Optional[Command]]:
            """Auxiliary recursive function."""
            if out is None:
                out = OrderedCommandDict(spaces_per_tab=spaces_per_tab)
            try:
                while (line := next(lines)) is not None:
                    match = re.match(r"(\s*)(\S+)(\s.*)?", line)
                    if match is None:
                        continue  # skip empty line
                    indent_str, command_name, command_argument = match.groups()
                    if command_argument is not None:
                        command_argument = command_argument[1:]
                    if any(ord(_) > 128 for _ in command_name) or not 1 <= len(command_name) < 256:
                        raise ValueError("Invalid command detected.")
                    if '\t' not in indent_str:
                        indent = len(indent_str)
                    else:  # Replace tabs with spaces before counting indentation
                        indent = sum((spaces_per_tab - (_ % spaces_per_tab)) if c == '\t' else 1 for _, c in enumerate(indent_str))
                    next_command = Command(name=command_name, argument=command_argument)
                    if indent <= parent_indent:
                        return out, next_command  # pass next command to
                    elif parent_indent < indent <= section_indent:  # be very forgiving
                        out.append(next_command)
                    else:  # indent > section_indent:  recurse
                        out[-1].children, next_command = parse_section(
                            lines,
                            parent_indent=section_indent, section_indent=indent, spaces_per_tab=2,
                            out=OrderedCommandDict([next_command], spaces_per_tab=spaces_per_tab))
                        if next_command is not None:
                            out.append(next_command)
            except StopIteration:
                pass
            return out, None

        return parse_section(file_contents.splitlines().__iter__(), spaces_per_tab=spaces_per_tab)[0]

    @classmethod
    def from_file(cls, input_path_or_stream: Union[FileLike, PathLike], spaces_per_tab: int = 2) -> OrderedCommandDict:
        """
        Reads a file into an `OrderedCommandDict` representation.

        :param input_path_or_stream: The file to read from, or its file-path.
        :param spaces_per_tab: The number of tabs to use when converting to str with __str__()

        :return: The OrderedCommandDict representation.
        """
        if isinstance(input_path_or_stream, PathLike):
            input_file = open(input_path_or_stream, 'rb')
        else:
            input_file = input_path_or_stream
        with input_file:
            contents = input_file.read()
            if len(contents) == 0:
                raise EOFError(f"Empty file: 0 bytes read from {input_file.name}.")
            if not isinstance(contents, str):
                """Attempt to parse the byte contents of the file using different encodings."""
                commands = None
                for encoding in ('utf-16', 'utf-16-le', 'utf-8', 'utf-8-sig', 'iso-8859-1'):
                    try:
                        commands = cls.from_str(contents.decode(encoding), spaces_per_tab=spaces_per_tab)
                    except UnicodeDecodeError as err:
                        log.debug(f"{err}: File {input_file.name} not encoded using {encoding}!")
                    except ValueError as err:
                        log.debug(f"{err}: File {input_file.name} does not seem to be encoded using {encoding}!")
                if commands is None:
                    raise IOError(f"Could not read {input_file.name}.")
        return commands

    @property
    def names(self) -> Sequence[str]:
        """A sequence of all the command names in order and potentially repeating."""
        return [_.name for _ in self.__commands]

    def __iter__(self) -> Iterator[Command]:
        """
        An iterator over all the commands in order.
        This is called when using
        ```
        for _ in command_dict:
            ...
        ```
        """
        return self.__commands.__iter__()

    def __len__(self) -> int:
        """The total number of commands."""
        return len(self.__commands)

    def __sizeof__(self) -> int:
        """The total number of commands."""
        return len(self)

    def __contains__(self, name_or_command: str | Command) -> bool:
        """
        Checks whether this contains a specific command or a command with a specific name.

        :param name_or_command: The name or command to look for.

        :return: True if the command is contained, False otherwise.
        """
        if isinstance(name_or_command, str):
            return name_or_command in self.names
        return name_or_command in self.__commands

    def __getitem__(self, item: int | Tuple | slice | str) -> Command | OrderedCommandDict:
        """
        Returns a command at a specific integer index or
        returns a `CommandSequence` with all commands with the specified name.
        """
        if not isinstance(item, Tuple):
            item = (item, )
        current_index = item[0]
        other_indices = item[1:]
        if isinstance(current_index, int):
            if current_index < len(self.__commands):
                result = self.__commands[current_index]
            else:
                raise IndexError(f"Only {len(self.__commands)} available. Index {current_index} does not exist.")
        elif isinstance(current_index, slice):
            result = OrderedCommandDict(self.__commands[current_index])
        else:  # isinstance(item, str)
            result = OrderedCommandDict([c for _, c in enumerate(self.__commands) if c.name == current_index])
        if len(item) > 1:
            result = result[other_indices]
        return result

    def sort_and_merge(self, name: str) -> OrderedCommandDict:
        """
        Get a sorted collection of arguments for an indexed command. Returns a new command dictionary for that is stored
        on the integer index at the start of their argument. The contents of identical indices are merged so that the
        returned collection does not contain duplicate indices.

        :param name: The command name to single out.

        :return: A collection of commands, sorted by its index, and merging items with the same index.
        """
        dict_of_lists = defaultdict(lambda: list[Command]())
        for c in self.__commands:
            if c.name == name:
                dict_of_lists[int(c.argument.split(maxsplit=1)[0])].append(c)

        def merge(commmands: Sequence[Command]) -> Command:
            """Auxiliary function to merge similar `Command`s into a single `Command`."""
            assert all(_.name == commmands[0].name for _ in commmands[1:]), "Can only merge commands with the same name."
            argument_matches = [re.match(r"\s*(\S+)(\s.*)", _.argument) for _ in commmands]
            arguments = [(_.groups()[1][1:] if _ is not None else "") for _ in argument_matches]
            children = list[Command]()
            for _ in commmands:
                if _.children is not None:
                    children += _.children
            return Command(name=commmands[0].name,
                           argument='\n'.join(arguments),
                           children=OrderedCommandDict(children))

        return OrderedCommandDict([merge(dict_of_lists[_]) for _ in sorted(dict_of_lists)])

    def __delitem__(self, index):
        del self.__commands[index]

    def append(self, new_command: Command) -> Self:
        """Adds a new command at the end of the sequence."""
        self.__commands.append(new_command)
        return self

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return '\n'.join(str(_) for _ in self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[repr(_) for _ in self]}, spaces_per_tab={self.spaces_per_tab})"


class Command:
    def __init__(self, name: str = "", argument: Optional[str] = None, children: Optional[OrderedCommandDict] = None):
        """
        Construct a new command, corresponding to a line or section of a text file.

        :param name: The command name, must be an ASCII string without whitespace.
        :param argument: The optional arguments, may start with an index number.
        :param children: An optional sub-tree of sub-commands.
        """
        self.name: str = name
        self.argument: Optional[str] = argument
        self.children: Optional[OrderedCommandDict] = children

    @property
    def words(self) -> Sequence[str]:
        """
        All the words in the argument.

        A word is defined as:

        * a string of characters without whitespace

        * a single or double-quoted string of characters on the same line.

        * a triple-quoted string of characters on multiple lines.
        """
        result = list[str]()
        if self.argument is not None:
            for match in re.finditer(r"'''([^']*)'''|'([^'\n\r]*)'|\"\"\"([^\"]*)\"\"\"|\"([^\"\n\r]*)\"|(\S+)",
                                     self.argument):
                result += [_ for _ in match.groups() if _ is not None]
        return result

    @property
    def numbers(self) -> Sequence[float]:
        """All the numbers in the argument."""
        result = list[float]()
        for w in self.words:
            try:
                result.append(float(w.strip(",;:\"\'")))
            except ValueError:
                pass
        return result

    def __getitem__(self, item: int | Tuple | slice | str) -> Command | OrderedCommandDict:
        """
        Returns a

        :param item: The selection index
        :return: A `Command` with the selected when item is an int, otherwise an `OrderedCommandDict` with the selection.
        """
        return self.children[item]

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        if self.argument is not None:
            result = ' '.join([self.name, self.argument])
        else:
            result = self.name
        if len(self.children) > 0:
            result += ''.join(f"\n{' ' * self.children.spaces_per_tab}{_}" for _ in self.children)
        return result

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}", "{self.argument}", {repr(self.children)})'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Command) -> bool:
        return hash(self) == hash(other)


class ZmxOpticalDesign(OpticalDesign):
    @staticmethod
    def from_str(contents: str, spaces_per_tab: int = 2) -> ZmxOpticalDesign:
        """Parses the text extracted from a .zmx file into an `OpticalDesign`."""
        return ZmxOpticalDesign(OrderedCommandDict.from_str(contents, spaces_per_tab))

    @staticmethod
    def from_file(input_path_or_stream: Union[FileLike, PathLike], spaces_per_tab: int = 2) -> ZmxOpticalDesign:
        """
        Reads a zmx file into an `OpticalDesign` representation.

        :param input_path_or_stream: The file to read the optical system from, or its file-path.
        :param spaces_per_tab: The optional number of spaces per tab.

        :return: A representation of the optical system.
        """
        return ZmxOpticalDesign(OrderedCommandDict.from_file(input_path_or_stream, spaces_per_tab))

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
            self.numerical_aperture = 2.0 / f_number  # todo: account for refractive index of object or image space?

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
        self.curvature = self.commands["CURV", 0].numbers[0] if "CURV" in self.commands else 0.0
        self.coating = self.commands["COAT", 0].argument if "COAT" in self.commands else ""
        self.radius = self.commands["DIAM", 0].numbers[0] / 2.0 if "DIAM" in self.commands else math.inf
        self.stop = "STOP" in self.commands
        self.distance = self.commands["DISZ", 0].numbers[0] if "DISZ" in self.commands else math.inf
        self.comment = self.commands["COMM", 0].argument if "COMM" in self.commands else ""
        glass_name = self.commands["GLAS", 0].words[0] if "GLAS" in self.commands else ""
        self.material = Material(name=glass_name)
        self.reflect = glass_name == "MIRROR"  # Not "MIRR" command for some reason
        # self.floating_aperture = self.commands["FLAP", 0].numbers if "FLAP" in self.commands else 0.0
        # self.conic_constant = self.commands["CONI", 0].numbers if "CONI" in self.commands else 0.0

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"
