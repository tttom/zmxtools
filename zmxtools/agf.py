from __future__ import annotations

import math
from typing import List, Dict, Union, Sequence, Tuple, Optional
from dataclasses import dataclass
import re
import itertools

from zmxtools.definitions import FileLike, PathLike, MaterialLibrary, Material
from zmxtools import log

log = log.getChild(__name__)

__all__ = ["AgfMaterialLibrary", "AgfMaterial"]


@dataclass
class Command:
    name: str
    argument: str
    children: CommandListDictDict
    line: str


@dataclass
class IndexedCommand(Command):
    index: int


class CommandListDict(Dict[int, List[Command]]):
    pass


class CommandListDictDict(Dict[str, CommandListDict | List[Command]]):
    pass


def _first_word(_: Command) -> str:
    return _.argument.split(" ", maxsplit=1)[0].strip()


def _as_numbers(c: Command) -> Sequence[float]:
    numbers_str = c.argument.split(" ")
    numbers = []
    for _ in numbers_str:
        try:
            numbers.append(float(_))
        except ValueError:
            pass
    return numbers


def _as_seq(cld: CommandListDict) -> Sequence[Sequence[Command]]:
    return [v for k, v in sorted(cld.items(), key=lambda _: _[0])]


def _as_flat_seq(cld: CommandListDict) -> Sequence[Command]:
    return tuple(itertools.chain(*_as_seq(cld)))


def _seq_as_str(cl: Sequence[Command]) -> str:
    return "\n".join(_.argument for _ in cl)


class AgfMaterialLibrary(MaterialLibrary):
    commands: CommandListDictDict

    @staticmethod
    def from_str(contents: str, spaces_per_tab: int = 2) -> AgfMaterialLibrary:
        """Parses the text extracted from a .agf file into an `MaterialLibrary` representation."""
        indexed_command_names = ["NOTE", "WAVM", "SURF", "PARM", "XDAT"]  # todo

        def parse_section(lines: Sequence[str], parent_indent: int = -spaces_per_tab, section_indent: Optional[int] = None
                          ) -> Tuple[CommandListDictDict, Sequence[str]]:
            command_list_dict_dict = CommandListDictDict()
            previous_command: Optional[Command] = None
            while len(lines) > 0:
                line = lines[0]
                # Replace tabs with spaces
                line = ''.join(' ' * (spaces_per_tab - (_ % spaces_per_tab)) if c == '\t' else c for _, c in enumerate(line))
                # break up line
                match = re.match(r"(\s*)(\S+)\s?(.*)", line)
                if match is None:
                    lines = lines[1:]
                    continue  # skip empty line
                indent_str, name, argument = match.groups()
                indent = len(indent_str)
                if section_indent is None:
                    section_indent = indent

                if not all(ord(c) < 256 for c in name):
                    raise ValueError(f"Non ascii command name {name} detected. Likely incorrect encoding used.")

                if parent_indent < indent <= section_indent:  # same section, add command to list
                    lines = lines[1:]
                    # Create either an IndexedCommand or a regular Command
                    if name in indexed_command_names:
                        index_str, argument = re.match(r"\s*(\d+)\s?(.*)", argument).groups()
                        index = int(index_str)
                        command = IndexedCommand(name=name, index=index, argument=argument, children=CommandListDictDict(), line=line)
                        if command.name not in command_list_dict_dict:
                            command_list_dict_dict[command.name] = CommandListDict()
                        if command.index not in command_list_dict_dict[command.name]:
                            command_list_dict_dict[command.name][command.index] = list[Command]()
                        command_list_dict_dict[command.name][command.index].append(command)
                    else:
                        command = Command(name=name, argument=argument, children=CommandListDictDict(), line=line)
                        if command.name not in command_list_dict_dict:
                            command_list_dict_dict[command.name] = list[Command]()
                        if command.name == "TYPE" and section_indent == 0:
                            log.error(f"Adding {command} with section_indent {section_indent}")
                        command_list_dict_dict[command.name].append(command)
                    previous_command = command
                elif indent <= parent_indent:
                    break  # This should not be added to this list, rather to the parent's list
                else:  # current_indent < indent
                    # Recurse and add children to the previous command
                    previous_command.children, lines = parse_section(lines, parent_indent=section_indent, section_indent=indent)

            return command_list_dict_dict, lines

        # Get the commands in a dictionary
        lines = contents.splitlines()
        command_dictionary, _ = parse_section(lines)

        return AgfMaterialLibrary(command_dictionary)

    @staticmethod
    def from_file(input_path_or_stream: Union[FileLike, PathLike]) -> MaterialLibrary:
        """
        Reads an .agf glass catalog file as an `MaterialLibrary`.

        :param input_path_or_stream: The file to read the material library from, or its file-path.

        :return: A representation of the material library.
        """
        if isinstance(input_path_or_stream, PathLike):
            input_file = open(input_path_or_stream, 'rb')
        else:
            input_file = input_path_or_stream
        with input_file:
            contents = input_file.read()
            if len(contents) == 0:
                raise EOFError(f"Empty .zmx file: 0 bytes read from {input_file.name}.")
            if not isinstance(contents, str):
                """Attempt to parse the byte contents of a .agf file using different encodings."""
                optical_model = None
                for encoding in ('utf-16', 'utf-16-le', 'utf-8', 'utf-8-sig', 'iso-8859-1'):
                    try:
                        optical_model = AgfMaterialLibrary.from_str(contents.decode(encoding))
                        if len(optical_model.surfaces) > 0:
                            log.debug(f"File {input_file.name} seems to be encoded using {encoding}.")
                            return optical_model
                        log.debug(f"No surfaces found in {input_file.name}! This file does not seem to be encoded using {encoding}!")
                    except UnicodeDecodeError as err:
                        log.debug(f"{err}: File {input_file.name} not encoded using {encoding}!")
                    except ValueError as err:
                        log.debug(f"{err}: File {input_file.name} does not seem to be encoded using {encoding}!")
                if optical_model is None:
                    raise IOError(f"Could not parse {input_file.name} as an optical model.")

    def __init__(self, commands: CommandListDictDict):
        self.commands = commands

        name = ""
        description = ""
        materials = []

        for k, v in self.commands.items():
            match k:
                case "VERS":
                    self.version = _seq_as_str(v)
                case "MODE":
                    mode = _first_word(v[-1])
                    if mode != "SEQ":
                        if mode == "NSC":
                            log.warning(f"Non-sequential mode not implemented.")
                        else:
                            log.warning(f"Unrecognized mode {v[-1].argument}.")
                        self.sequential = False
                case "NAME":
                    name = _seq_as_str(v)
                case "COMM":
                    description = _seq_as_str(v)
                case "MAT":
                    self.surfaces = [AgfMaterial(s.children) for s in _as_flat_seq(v)]
                case _:
                    if k not in ["ENPD", "ENVD", "GFAC", ]:
                        log.warning(f"Unrecognized glass catalog property {k} {v}")

        super().__init__(name=name, description=description, materials=materials)

    def __str__(self) -> str:
        descriptors = dict(name=self.name, note=self.description)
        return (f"""{self.__class__.__name__}({''.join(f'{k}="{v}", ' for k, v in descriptors.items() if len(v) > 0)}"""
                f"materials={self.surfaces})")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"


class AgfMaterial(Material):
    """A class to represent a Zemax glass"""
    commands: CommandListDictDict

    def __init__(self, commands: CommandListDictDict):
        self.commands = commands

        name = ""

        for k, v in self.commands.items():
            match k:
                case "TYPE":
                    self.type = v[-1].argument
                case _:
                    if k not in ["FIMP", "HIDE", "MEMA", "POPS",]:
                        log.warning(f"Unrecognized surface property {k} {v}")

        super().__init__(name)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.name)})"


