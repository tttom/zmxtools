import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Union
from lark import Lark, tree, lexer, Transformer

from . import log

log = log.getChild(__name__)

__all__ = ["Material", "MaterialLibrary", "read"]


__GRAMMAR = r"""
    material_library: WS* (descriptor NEWLINE)+
    
    ?descriptor: version | mode | name
    
    ?number_or_infinity: NUMBER | "INFINITY" -> infinity
    
    STRING: /.+/
    
    %import common.INT
    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER    -> NUMBER
    %import common.NEWLINE
    %import common.WS
    %import common.WS_INLINE
    %ignore WS_INLINE
"""


@dataclass
class Material:
    name: str


@dataclass
class MaterialLibrary:
    name: str = ""
    description: str = ""
    materials: Sequence[Material] = field(default_factory=list)


class AgfTransformer(Transformer):
    def material_library(self, descriptors) -> MaterialLibrary:
        pass


def __parse_agf_str(contents: str) -> MaterialLibrary:
    """Parse the str contents of a .agf file."""
    parser = Lark(__GRAMMAR, start='material_library')
    parse_tree = parser.parse(contents)
    material_library = AgfTransformer().transform(parse_tree)
    return material_library


def __parse_agf_bytes(content_bytes: bytes) -> MaterialLibrary:
    """Attempt to parse the byte contents of a .agf file using different encodings."""
    for encoding in ('utf-16', 'utf-16-le', 'utf-8', 'utf-8-sig', 'iso-8859-1'):
        try:
            contents = content_bytes.decode(encoding)
            return __parse_agf_str(contents)
        except UnicodeDecodeError as err:
            pass


def read(input_file_or_path: Union[io.IOBase, io.BytesIO, Path, str]) -> MaterialLibrary:
    """
    Reads a .agf Zemax Glass file.

    :param input_file_or_path: The file to read the material library from, or its file-path.

    :return: A representation of the material library.
    """
    if isinstance(input_file_or_path, Union[io.IOBase, io.BytesIO]):
        content_bytes = input_file_or_path.read()
    else:
        with open(input_file_or_path, 'rb') as input_file:
            content_bytes = input_file.read()
    return __parse_agf_bytes(content_bytes)

