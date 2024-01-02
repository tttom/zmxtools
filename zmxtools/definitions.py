from __future__ import  annotations

import typing
from typing import Optional, Self, Sequence
import pathlib
import io
import os
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
