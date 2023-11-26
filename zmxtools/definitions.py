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


@dataclass
class Material:
    """A class to represent a Zemax glass"""
    name: str

vacuum = Material("vacuum")


@dataclass
class MaterialLibrary:
    """A class to represent a collection of :class:`Material`s."""
    name: str = ""
    description: str = ""
    materials: Sequence[Material] = field(default_factory=list)


@dataclass
class Surface:
    """A class to represent a thin surface between two volumes."""
    index: int = -1
    stop: bool = False
    curvature: float = 0.0
    radius: float = 0.0
    distance: float = 0.0
    name: str = ""  # The common name of the lens element
    glass: Material = field(default_factory=lambda: vacuum)


@dataclass
class OpticalSystem:
    """A class to describe a complete optical system, including light source and detector surface."""
    version: str = ""
    name: str = ""
    description: str = ""
    surfaces: List[Surface] = field(default_factory=list)
    unit: float = 1.0
    wavelengths: Sequence[float] = field(default_factory=list)
    wavelength_weights: Sequence[float] = field(default_factory=list)
    material_library: MaterialLibrary = field(default_factory=lambda: MaterialLibrary())
    coating_filename: str = ""
