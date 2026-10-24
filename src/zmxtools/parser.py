from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Self

import zmxtools
from zmxtools.utils.io import FileLike, PathLike

log = zmxtools.config_logger(__name__)


class OrderedCommandDict:
    """An ordered dictionary of commands, as they appear in the file, but indexable by the command name."""

    def __init__(self, commands: Sequence[Command] = (), spaces_per_indent: int = 2):
        """
        Construct a new command dictionary from a sequence of commands.

        Args:
            commands: A sequence of commands, which will be kept in order.
            spaces_per_indent: The number of spaces to use when indenting with __str__()

        """
        assert all(_ is not None for _ in commands), (
            f"{self.__class__.__name__} expected as sequence of Commands, not {commands}."
        )
        self.__commands = list(commands)
        self.spaces_per_indent: int = spaces_per_indent  # For __str__()

    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 2) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text.

        Args:
            file_contents: The text string extracted from an optical file.
            spaces_per_indent: The number of spaces per indent to assume

        Returns:
            The command dictionary.

        """
        raise NotImplementedError

    @classmethod
    def from_file(
        cls,
        input_path_or_stream: FileLike | PathLike,
        spaces_per_indent: int = 2,
        encoding: str = "utf-16",
    ) -> OrderedCommandDict:
        """
        Reads a file into an `OrderedCommandDict` representation.

        Args:
            input_path_or_stream: The file to read from, or its file-path.
            spaces_per_indent: The number of spaces to use when indenting with __str__()
            encoding: The text-encoding to try first.

        Returns:
            The OrderedCommandDict representation.

        """
        encodings = (
            encoding,
            *(_ for _ in ("utf-16", "utf-8-sig", "utf-16-le", "utf-8", "utf-16-be", "iso-8859-1") if _ != encoding),
        )
        with (
            open(input_path_or_stream, "rb") if isinstance(input_path_or_stream, PathLike) else input_path_or_stream
        ) as input_file:
            contents = input_file.read()
            if len(contents) == 0:
                raise EOFError(f"Empty file: 0 bytes read from {input_file.name}.")
            if isinstance(contents, str):
                return cls.from_str(contents, spaces_per_indent=spaces_per_indent)
            # Attempt to parse the byte-contents of the file using different encodings.
            commands = None
            for encoding_to_try in encodings:
                try:
                    commands = cls.from_str(contents.decode(encoding_to_try), spaces_per_indent=spaces_per_indent)

                    def is_ascii(name: str) -> bool:
                        return all((_ <= 128 and 0x30 <= ord(c) < 0x7F) for _, c in enumerate(name))

                    if all(is_ascii(name) for name in commands.names):
                        log.debug(f"File {input_file.name} appears to be encoded using {encoding_to_try}.")
                        break
                    log.debug(
                        "Strange command detected. "
                        + f"File {input_file.name} does not seem to be encoded using {encoding_to_try}!",
                    )
                except UnicodeError as err:
                    log.debug(f"File {input_file.name} not encoded using {encoding_to_try}! UnicodeError: {err}")
            if commands is not None:
                return commands
            raise OSError(f"Could not decode {input_file.name} with either of {encodings}.")

    @property
    def names(self) -> Sequence[str]:
        """A sequence of all the command names in order and potentially repeating."""
        return [_.name for _ in self.__commands]

    def __iter__(self) -> Iterator[Command]:
        """
        An iterator over all the commands in order.

        This is called when using a for loop as

        .. code:: python

            for _ in command_dict:
                ...
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

        Args:
            name_or_command: The name or command to look for.

        Returns:
            True if the command is contained, False otherwise.

        """
        if isinstance(name_or_command, str):
            return name_or_command in self.names
        return name_or_command in self.__commands

    def __getitem__(self, item: int | slice | str | tuple[int | slice | str, ...]) -> Command | OrderedCommandDict:
        """
        Returns a command at a specific integer index or a `CommandSequence` with all commands with the specified name.
        """
        if not isinstance(item, tuple):
            item = (item,)
        current_index = item[0]
        other_indices = item[1:]
        if isinstance(current_index, int):
            if current_index >= len(self.__commands):
                raise IndexError(f"Only {len(self.__commands)} available. Index {current_index} does not exist.")
            result = self.__commands[current_index]
        elif isinstance(current_index, slice):
            result = OrderedCommandDict(self.__commands[current_index])
        else:  # isinstance(item, str)
            result = OrderedCommandDict([c for _, c in enumerate(self.__commands) if c.name == current_index])
        if len(item) > 1:
            result = result[other_indices]
        return result

    def command(self, name: str, index: int = 0) -> Command:
        """
        Return the single :py:class:`Command` stored under ``name`` at ``index``.

        This is a typed convenience wrapper around ``self[name, index]`` that always yields a ``Command``.

        Args:
            name: The command name to look up.
            index: The index within the commands sharing that name.

        Returns:
            The matching ``Command``.

        """
        result = self[name]
        if isinstance(result, OrderedCommandDict):
            result = result[index]
        assert isinstance(result, Command), f"Expected a Command at {name!r}[{index}]."
        return result

    def commands_of(self, name: str) -> OrderedCommandDict:
        """
        Return all commands sharing ``name`` as a typed :py:class:`OrderedCommandDict`.

        Args:
            name: The command name to look up.

        Returns:
            The ordered collection of commands with that name.

        """
        result = self[name]
        assert isinstance(result, OrderedCommandDict), f"Expected an OrderedCommandDict at {name!r}."
        return result

    def sort_and_merge(self, name: str) -> OrderedCommandDict:
        """
        Get a sorted collection of arguments for an indexed command.

        This method returns a new command dictionary for that is stored on the integer index at the start of their
        argument. The contents of identical indices are merged so that the returned collection does not contain
        duplicate indices.

        Args:
            name: The command name to single out.

        Returns:
            A collection of commands, sorted by its index, and merging items with the same index.

        """
        dict_of_lists = defaultdict(lambda: list[Command]())
        for c in self.__commands:
            if c.name == name and c.argument is not None:
                index_str = c.argument.split(maxsplit=1)[0]
                try:
                    index = int(index_str)
                except ValueError:  # Skip commands that are not indexed by an integer
                    continue
                dict_of_lists[index].append(c)

        def merge(commmands: Sequence[Command]) -> Command:
            """Auxiliary function to merge similar `Command`s into a single `Command`."""
            assert all(_.name == commmands[0].name for _ in commmands[1:]), (
                "Can only merge commands with the same name."
            )
            argument_matches = [re.match(r"\s*(\S+)(\s.*)", _.argument or "") for _ in commmands]
            arguments = [
                (m[1][1:] if m is not None and m[1] is not None else "")
                for m in (_.groups() if _ is not None else None for _ in argument_matches)
            ]
            children = list[Command]()
            for _ in commmands:
                if _.children is not None:
                    children += _.children
            return Command(
                name=commmands[0].name,
                argument="\n".join(arguments),
                children=OrderedCommandDict(children),
            )

        return OrderedCommandDict([merge(dict_of_lists[_]) for _ in sorted(dict_of_lists)])

    def append(self, new_command: Command) -> Self:
        """Adds a new command at the end of the sequence."""
        self.__commands.append(new_command)
        return self

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.

        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return "\n".join(str(_) for _ in self)

    def __repr__(self) -> str:
        """Returns a string that fully represents this object."""
        return f"{self.__class__.__name__}({list(self)}, spaces_per_indent={self.spaces_per_indent})"


class Command:
    """A class to represent a single command/line in a file."""

    def __init__(self, name: str = "", argument: str | None = None, children: OrderedCommandDict | None = None):
        """
        Construct a new command, corresponding to a line or section of a text file.

        Args:
            name: The command name, must be an ASCII string without whitespace.
            argument: The optional arguments, may start with an index number.
            children: An optional sub-tree of sub-commands.

        """
        self.name: str = name
        self.argument: str | None = argument
        self.children: OrderedCommandDict | None = children

    @property
    def text(self) -> str:
        """The argument as a plain string, or an empty string when there is no argument."""
        return "" if self.argument is None else self.argument

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
            word_pattern = re.compile(r"'''([^']*)'''|'([^'\n\r]*)'|\"\"\"([^\"]*)\"\"\"|\"([^\"\n\r]*)\"|(\S+)")
            for match in word_pattern.finditer(self.argument):
                for _ in match.groups():
                    if _ is not None:
                        result.append(_)
        return result

    @property
    def numbers(self) -> Sequence[float]:
        """All the numbers in the argument."""
        result = list[float]()
        for w in self.words:
            try:
                result.append(float(w.strip(",;:\"'")))
            except ValueError:
                pass
        return result

    def __contains__(self, name_or_command: str | Command) -> bool:
        """
        Checks whether this contains a specific sub-command or a sub-command with a specific name.

        Args:
            name_or_command: The name or command to look for.

        Returns:
            True if the command is contained, False otherwise.

        """
        return self.children is not None and name_or_command in self.children

    def __len__(self) -> int:
        """
        The number of sub-commands.
        """
        return 0 if self.children is None else len(self.children)

    def __getitem__(self, item: int | slice | str | tuple[int | slice | str, ...]) -> Command | OrderedCommandDict:
        """
        Returns the sub-command with a specific name, index or a collection of sub-commands at a multiple indices.

        Args:
            item: The selection index, name, or indices.

        Returns:
            A :py:class:`Command` with the selected when item is an int, otherwise an :py:class:`OrderedCommandDict` with the selection.

        """
        if self.children is None:
            raise IndexError(f"Command {self.name!r} has no sub-commands to index with {item!r}.")
        return self.children[item]

    def command(self, name: str, index: int = 0) -> Command:
        """
        Return the single child :py:class:`Command` stored under ``name`` at ``index``.

        Args:
            name: The sub-command name to look up.
            index: The index within the sub-commands sharing that name.

        Returns:
            The matching ``Command``.

        """
        result = self[name]
        if isinstance(result, OrderedCommandDict):
            result = result[index]
        assert isinstance(result, Command), f"Expected a Command at {name!r}[{index}]."
        return result

    def commands_of(self, name: str) -> OrderedCommandDict:
        """
        Return all child commands sharing ``name`` as a typed :py:class:`OrderedCommandDict`.

        Args:
            name: The sub-command name to look up.

        Returns:
            The ordered collection of sub-commands with that name.

        """
        result = self[name]
        assert isinstance(result, OrderedCommandDict), f"Expected an OrderedCommandDict at {name!r}."
        return result

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
            result = " ".join([self.name, self.argument])
        else:
            result = self.name
        if self.children is not None and len(self.children) > 0:
            result += "".join(f"\n{' ' * self.children.spaces_per_indent}{_}" for _ in self.children)
        return result

    def __repr__(self) -> str:
        """Returns a complete string representation of this object."""
        return f'{self.__class__.__name__}("{self.name}", "{self.argument}", {self.children!r})'
