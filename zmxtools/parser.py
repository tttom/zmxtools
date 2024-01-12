from __future__ import  annotations

import re
from typing import Optional, Self, Sequence, Union, Iterator, Tuple
from collections import defaultdict

from zmxtools.utils.io import FileLike, PathLike
from zmxtools import log

log = log.getChild(__name__)


class OrderedCommandDict:
    def __init__(self, commands: Sequence[Command] = tuple(), spaces_per_indent: int = 2):
        """
        Construct a new command dictionary from a sequence of commands.

        :param commands: A sequence of commands, which will be kept in order.
        :param spaces_per_indent: The number of spaces to use when indenting with __str__()
        """
        assert all(_ is not None for _ in commands), f"{self.__class__.__name__} expected as sequence of Commands, not {commands}."
        self.__commands = list(commands)
        self.spaces_per_indent: int = spaces_per_indent  # For __str__()

    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 2) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text.

        :param file_contents: The text string extracted from an optical file.
        :param spaces_per_indent: The number of spaces per indent to assume

        :return: The command dictionary.
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, input_path_or_stream: Union[FileLike, PathLike],
                  spaces_per_indent: int = 2,
                  encoding: str = 'utf-16') -> OrderedCommandDict:
        """
        Reads a file into an `OrderedCommandDict` representation.

        :param input_path_or_stream: The file to read from, or its file-path.
        :param spaces_per_indent: The number of spaces to use when indenting with __str__()
        :param encoding: The text-encoding to try first.

        :return: The OrderedCommandDict representation.
        """
        if isinstance(input_path_or_stream, PathLike):
            input_file = open(input_path_or_stream, 'rb')
        else:
            input_file = input_path_or_stream
        encodings = ('utf-16', 'utf-8-sig', 'utf-16-le', 'utf-8', 'utf-16-be', 'iso-8859-1')
        if encoding not in encodings:
            encodings = (encoding, *encodings)
        with input_file:
            contents = input_file.read()
            if len(contents) == 0:
                raise EOFError(f"Empty file: 0 bytes read from {input_file.name}.")
            if not isinstance(contents, str):
                """Attempt to parse the byte-contents of the file using different encodings."""
                commands = None
                for encoding_to_try in encodings:
                    try:
                        commands = cls.from_str(contents.decode(encoding_to_try), spaces_per_indent=spaces_per_indent)
                        if all(all((_ <= 128 and 0x30 <= ord(c) < 0x7f) for _, c in enumerate(name)) for name in commands.names):
                            log.debug(f"File {input_file.name} appears to be encoded using {encoding_to_try}.")
                            break
                        log.debug(f'Strange command detected. File {input_file.name} does not seem to be encoded using {encoding_to_try}!')
                    except UnicodeError as err:
                        log.debug(f"File {input_file.name} not encoded using {encoding_to_try}! UnicodeError: {err}")
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

    def __delitem__(self, index: int):
        """Delete the value at the specified index."""
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
        return f"{self.__class__.__name__}({list(self)}, spaces_per_indent={self.spaces_per_indent})"


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

    def __contains__(self, name_or_command: str | Command) -> bool:
        """
        Checks whether this contains a specific sub-command or a sub-command with a specific name.

        :param name_or_command: The name or command to look for.

        :return: True if the command is contained, False otherwise.
        """
        return name_or_command in self.children

    def __len__(self) -> int:
        """
        The number of sub-commands.
        """
        return len(self.children)

    def __getitem__(self, item: int | Tuple | slice | str) -> Command | OrderedCommandDict:
        """
        Returns the sub-command with a specific name, index or a collection of sub-commands at a multiple indices.

        :param item: The selection index, name, or indices.
        :return: A `Command` with the selected when item is an int, otherwise an `OrderedCommandDict` with the selection.
        """
        return self.children[item]

    def append(self, new_command: Command) -> Self:
        """Adds a new command at the end of the sequence."""
        if self.children is None:
            self.children = OrderedCommandDict()
        self.children.append(new_command)
        return self

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        if self.argument is not None:
            result = ' '.join([self.name, self.argument])
        else:
            result = self.name
        if self.children is not None and len(self.children) > 0:
            result += ''.join(f"\n{' ' * self.children.spaces_per_indent}{_}" for _ in self.children)
        return result

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}", "{self.argument}", {repr(self.children)})'

    # def __hash__(self) -> int:
    #     return hash(repr(self))
    #
    # def __eq__(self, other: Command) -> bool:
    #     return hash(self) == hash(other)
