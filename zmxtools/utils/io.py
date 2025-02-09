from __future__ import annotations

import io
import pathlib
import typing

import typing_extensions

from zmxtools import log

log = log.getChild(__name__)


class BytesFile:
    """A class to represent bytes as a file stream without it coming from disk."""

    def __init__(self, path: pathlib.Path | str, contents: typing.Optional[bytes] = None):
        """
        Construct a ByteFile representation from a path and file contents.

        :param path: The path of the file.
        :param contents: The file's contents.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if contents is None:
            with open(path, 'rb') as f:
                contents = f.read()
        self.__content_bytes: bytes = contents
        self.__content_stream: typing.Optional[io.BytesIO] = None
        self.__path: pathlib.Path = path

    @property
    def path(self) -> pathlib.Path:
        """The path of this file (can be fictitious)."""
        return self.__path

    @property
    def name(self) -> str:
        """The name of this file."""
        return self.path.as_posix()

    def open(self) -> typing.Self:
        """Open the file or stream."""
        if self.__content_stream is None:
            self.__content_stream = io.BytesIO(self.__content_bytes)
        return self

    def close(self) -> None:
        """Close the file or stream."""
        if self.__content_stream is not None:
            self.__content_stream.close()
        self.__content_stream = None

    def __enter__(self) -> typing.Self:
        """Called upon entering the context manager (using the with statement)."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> typing_extensions.Literal[False]:
        """Called upon exiting the context manager."""
        self.close()
        return False  # Do not suppress Exceptions

    def read(self, n: int = -1) -> bytes:
        """Read from the file, just like a typing.BinaryIO object."""
        if self.__content_stream is None:
            with self:
                return self.read(n)
        return self.__content_stream.read(n)


BinaryFileLike = BytesFile | typing.BinaryIO
FileLike = BinaryFileLike | typing.TextIO
PathLike = pathlib.Path | str
