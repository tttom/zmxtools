from __future__ import  annotations

import typing
from dataclasses import dataclass, field
import pathlib
import io
import os
from typing import Optional, Self, List, Sequence, Union


class BytesFile:
    """A class to represent bytes as a file stream without it coming from disk."""
    def __init__(self, path: pathlib.Path | str, contents: Optional[bytes] = None):
        if isinstance(path, str):
            path = pathlib.Path(path)
        if contents is None:
            with open(path, "rb") as f:
                contents = f.read()
        self.__content_bytes: bytes = contents
        self.__content_stream: io.BytesIO = None
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


BinaryFileLike = Union[BytesFile, typing.BinaryIO]
FileLike = Union[BinaryFileLike, typing.IO]
PathLike = Union[os.PathLike, str]


class Material:
    """A class to represent a material as glass"""
    name: str = ""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Material) -> bool:
        return hash(self) == hash(other)


class Vacuum(Material):
    def __init__(self):
        super().__init__("vacuum")


class MaterialLibrary:
    name: str = ""
    description: str = ""
    materials: Sequence[Material] = []

    def __init__(self, name: str, description: str = "", materials: Sequence[Material] = tuple()):
        self.name = name
        self.description = description
        self.materials = materials

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.description}, {repr(self.materials)})"


class Surface:
    """A class to represent a thin surface between two volumes."""
    stop: bool = False
    description: str = ""  # A comment, often the name of the lens element

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stop={self.stop}, description={self.description})"


class OpticalDesign:
    name: str = ""
    description: str = ""

    surfaces: List[Surface] = []
    wavelengths: Sequence[float] = []
    wavelength_weights: Sequence[float] = []
    material_libraries: Sequence[MaterialLibrary] = []

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.name}, {self.description}, {repr(self.surfaces)}, "
                f"wavelengths={self.wavelengths}, wavelength_weights={self.wavelength_weights}, material_library={repr(self.material_libraries)})")


