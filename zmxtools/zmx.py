import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Dict, Union
from collections import defaultdict
from lark import Lark, tree, lexer, Transformer

from .agf import Material
from . import log

log = log.getChild(__name__)

__all__ = ["OpticalSystem", "read"]


__GRAMMAR = r"""
    optical_system: WS* (descriptor NEWLINE)+
    
    ?descriptor: version | mode | name | note
        | pfil
        | unit
        | enpd | envd | gfac
        | glass_catalog
        | ray_aim
        | push
        | sdma | ftyp | ropd | picb
        | fln
        | fwgn
        | vdn
        | vcn
        | vann
        | wavelength
        | pwav
        | polarization
        | glrs | gstd | nscd
        | coating_filename
        | surface
        | tolerance
        | tcmm | mnum | moff
        | unknown_descriptor
    
    version: "VERS" STRING
    mode: "MODE" CNAME
    name: "NAME" " " STRING
    note: ("NOTE" INT " " STRING)+
    pfil: "PFIL" INT+
    unit: "UNIT" unit_symbol STRING*
    enpd: "ENPD" NUMBER
    envd: "ENVD" NUMBER INT INT
    gfac: "GFAC" INT INT
    glass_catalog: "GCAT" STRING
    ray_aim: "RAIM" NUMBER+
    push: "PUSH" INT+
    sdma: "SDMA" INT+
    ftyp: "FTYP" INT+
    ropd: "ROPD" INT
    picb: "PICB" INT
    fln: "XFLN" NUMBER+ NEWLINE "YFLN" NUMBER+
    fwgn: "FWGN" NUMBER+
    vdn: "VDXN" NUMBER+ NEWLINE "VDYN" NUMBER+
    vcn: "VCXN" NUMBER+ NEWLINE "VCYN" NUMBER+
    vann: "VANN" NUMBER+
    wavelength: "WAVM" INT NUMBER INT
    pwav: "PWAV" INT
    polarization: "POLS" INT+
    glrs: "GLRS" INT INT
    gstd: "GSTD" INT NUMBER+ INT+
    nscd: "NSCD" NUMBER+
    coating_filename: "COFN QF" ESCAPED_STRING* | "COFN" filename*
    surface: ( "SURF" INT NEWLINE (surface_property NEWLINE)+ )+ "BLNK"
    tolerance: "TOL" tolerance_type INT INT NUMBER NUMBER INT INT INT
    tcmm: "TCMM" STRING
    mnum: "MNUM" INT+
    moff: "MOFF" INT INT ESCAPED_STRING NUMBER+ ESCAPED_STRING
    unknown_descriptor.-1: STRING
        
    unit_symbol: "MM" | "CM" | "DM" | "M" | "IN" | CNAME
    ?surface_property: surface_type
            | surface_stop
            | surface_curvature
            | surface_mirror
            | surface_distance_z
            | surface_diameter
            | surface_hide | surface_slab | surface_pops | surface_flap
            | surface_coating
            | surface_glass
            | surface_unknown_property
                    
    tolerance_type: "COMP" | "TWAV" | "TRAD" | "TTHI" | "TSDX" | "TSDY" | "TSTX" | "TSTY" | "TIRR" | "TIND" | "TABB"
    filename: CNAME ("." CNAME)+
    
    surface_stop: "STOP"
    surface_type: "TYPE" " " STRING
    surface_curvature: "CURV" NUMBER+ ESCAPED_STRING
    surface_hide: "HIDE" NUMBER+
    surface_mirror: "MIRR" NUMBER NUMBER
    surface_slab: "SLAB" INT
    surface_distance_z: "DISZ" number_or_infinity 
    surface_diameter: "DIAM" NUMBER+ ESCAPED_STRING
    surface_pops: "POPS" NUMBER+
    surface_coating: "COAT" " " STRING
    surface_glass: "GLAS" " " STRING NUMBER*
    surface_flap: "FLAP" NUMBER+
    surface_unknown_property.-1: STRING
    
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
class Surface:
    index: int = -1
    stop: bool = False
    curvature: float = 0.0
    distance: float = 0.0
    radius: float = 0.0
    glass: Material = None


@dataclass
class OpticalSystem:
    version: str = ""
    name: str = ""
    description: str = ""
    surfaces: List[Surface] = field(default_factory=list)
    unit: float = 1.0


class ZmxTransformer(Transformer):
    def optical_system(self, descriptors) -> OpticalSystem:
        optical_system = OpticalSystem()

        notes: Dict[int, str] = defaultdict[int, str](str)

        for descriptor in descriptors:
            if isinstance(descriptor, Surface):
                log.error(f"Found {descriptor} Surface")
                optical_system.surfaces.append(descriptor)
            elif descriptor is not None and isinstance(descriptor, tree.Tree):
                match descriptor.data:
                    case "version":
                        optical_system.version = descriptor.children[0].value.strip()
                    case "mode":
                        mode_str = descriptor.children[0].value
                        is_sequential = mode_str.upper().startswith("SEQ")
                        if not is_sequential:
                            log.warning(f"This does not seem to be a sequential model because the MODE is specified as '{mode_str}'.")
                    case "note":
                        note_idx = int(descriptor.children[0].value)
                        notes[note_idx] += descriptor.children[1].value
                    case "name":
                        optical_system.name = descriptor.children[0].value
                    case "unit":
                        match descriptor.children[0].value.upper():
                            case "UM":
                                unit = 1e-6
                            case "MM":
                                unit = 1e-3
                            case "CM":
                                unit = 10e-3
                            case "DM":
                                unit = 100e-3
                            case unit_str if unit_str in ("IN" or "INCH"):
                                unit = 25.4e-3
                            case "FT":
                                unit = 304.8e-3
                            case _:
                                unit = 1.0
                        optical_system.unit = unit
                    case _:
                        log.error(f"Unknown descriptor: '{descriptor.data}' with values {descriptor.children}!")
            elif isinstance(descriptor, lexer.Token) and descriptor.value.strip() == "":
                pass  # Ignore white-space
            else:
                log.error(f"Not recognized as descriptor: _{descriptor.value.strip()}_")

        sorted_notes: List[str] = [v for k, v in sorted(notes.items(), key=lambda _: _[0])]
        optical_system.description = "\n".join(sorted_notes)

        return optical_system

    def surface(self, _) -> Surface:
        print(f"surface() {_}")
        surface = Surface()

        for surface_property in _:
            if isinstance(surface_property, tree.Tree):
                match surface_property.data:
                    case "surface_stop":
                        surface.stop = True
                    case "surface_type":
                        surface_type = surface_property.children[0]
                        if surface_type.upper() != "STANDARD":
                            log.warning(f"Unrecognized surface type '{surface_type}'")
                    case "surface_distance_z":
                        distance_z = surface_property.children[0]
                        if isinstance(distance_z, tree.Tree) and distance_z.data == "infinity":
                            distance_z = math.inf
                        surface.distance = float(distance_z) / 2.0
                    case "surface_curvature":
                        surface.curvature = float(surface_property.children[0])
                    case "surface_diameter":
                        surface.radius = float(surface_property.children[0]) / 2.0
                    case "surface_mirror":
                        mirror_type = surface_property.children[0]
                        if int(mirror_type) != 2:
                            log.warning(f"Unrecognized mirror type '{mirror_type}'")
                    case s_prop if s_prop in ("surface_hide", "surface_slab", "surface_pops", "surface_flap"):
                        log.debug(f"Ignoring surface property: '{surface_property.data}' with values {surface_property.children}!")
                    case _:
                        log.error(f"Unknown surface property: '{surface_property.data}' with values {surface_property.children}!")

        return surface


def __parse_zmx_str(contents: str) -> OpticalSystem:
    """Parse the str contents of a .zmx file."""
    parser = Lark(__GRAMMAR, start='optical_system')  #, ambiguity='explicit')
    parse_tree = parser.parse(contents)
    optical_system = ZmxTransformer().transform(parse_tree)
    return optical_system


def __parse_zmx_bytes(content_bytes: bytes) -> OpticalSystem:
    """Attempt to parse the byte contents of a .zmx file using different encodings."""
    for encoding in ('utf-16', 'utf-16-le', 'utf-8', 'utf-8-sig', 'iso-8859-1'):
        try:
            contents = content_bytes.decode(encoding)
            return __parse_zmx_str(contents)
        except UnicodeDecodeError as err:
            pass


def read(input_file_or_path: Union[io.IOBase, io.BytesIO, Path, str]) -> OpticalSystem:
    """
    Reads a zmx file.

    :param input_file_or_path: The file to read the optical system from, or its file-path.

    :return: A representation of the optical system.
    """
    if isinstance(input_file_or_path, Union[io.IOBase, io.BytesIO]):
        content_bytes = input_file_or_path.read()
    else:
        with open(input_file_or_path, 'rb') as input_file:
            content_bytes = input_file.read()
    return __parse_zmx_bytes(content_bytes)
