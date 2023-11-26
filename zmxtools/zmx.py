import math
from typing import List, Dict, Union
from collections import defaultdict
from lark import Lark, tree, lexer, Transformer

from zmxtools.definitions import Material, MaterialLibrary, Surface, OpticalSystem, FileLike, PathLike
from zmxtools import log

log = log.getChild(__name__)

__all__ = ["read", "OpticalSystem", "Surface"]


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
        | xfln | yfln | xfld | yfld
        | fwgn | fwgt
        | vcxn | vcyn
        | vdxn | vdyn
        | vann | zvan
        | zvcx | zvcy| zvdx | zvdy
        | wavelengths | wavelengths_n
        | wavelength_weights | wavelength_weights_n
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
    unit: "UNIT" WS+ unit_symbol WS+ STRING*
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
    xfln: "XFLN" NUMBER+
    yfln: "YFLN" NUMBER+
    xfld: "XFLD" NUMBER+
    yfld: "YFLD" NUMBER+
    fwgn: "FWGN" NUMBER+
    fwgt: "FWGT" NUMBER+
    vdxn: "VDXN" NUMBER+
    vdyn: "VDYN" NUMBER+
    vcxn: "VCXN" NUMBER+
    vcyn: "VCYN" NUMBER+
    vann: "VANN" NUMBER+
    zvan: "ZVAN" NUMBER+
    zvcx: "ZVCX" NUMBER+
    zvcy: "ZVCY" NUMBER+
    zvdx: "ZVDX" NUMBER+
    zvdy: "ZVDY" NUMBER+
    wavelengths: "WAVL" NUMBER+
    wavelengths_n: "WAVN" NUMBER+
    wavelength_weights: "WWGT" NUMBER+
    wavelength_weights_n: "WWGN" NUMBER+
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
            | surface_name
            | surface_curvature
            | surface_mirror
            | surface_distance_z
            | surface_diameter
            | surface_hide | surface_slab | surface_pops | surface_flap
            | surface_coating
            | surface_glass
            | surface_unknown_property

    tolerance_type: "COMP" | "TWAV" | "TRAD" | "TTHI" | "TSDX" | "TSDY" | "TSTX" | "TSTY" | "TIRR" | "TIND" | "TABB" | "TOFF"
    filename: CNAME ("." CNAME)+

    surface_stop: "STOP"
    surface_name: "COMM" " " STRING
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


class ZmxTransformer(Transformer):
    def optical_system(self, descriptors) -> OpticalSystem:
        optical_system = OpticalSystem()

        notes: Dict[int, str] = defaultdict[int, str](str)

        for descriptor in descriptors:
            if isinstance(descriptor, Surface):
                log.error(f"Found {descriptor}")
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
                        match descriptor.children[0].value.strip().upper():
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
                    case "glass_catalog":
                        optical_system.material_library = MaterialLibrary(descriptor.children[0].value.strip())
                    case "coating_filename":
                        optical_system.coating_filename = ".".join(_.value.strip() for _ in descriptor.children[0].children)
                    case "wavelengths":
                        optical_system.wavelengths = [float(_.value.strip()) for _ in descriptor.children]
                    case "wavelength_weights":
                        optical_system.wavelength_weights = [float(_.value.strip()) for _ in descriptor.children]
                    case desc if desc in ("enpd", "envd", "gfac", "ray_aim", "push", "sdma", "ftyp", "ropd", "picb",
                                          "pwav", "glrs", "gstd", "nscd", "fwgn", "mnum", "moff",
                                          "xfld", "yfld", "xfln", "yfln", "vcxn", "vcyn", "vdxn", "vdyn",
                                          "fwgt", "zvcx", "zvcy", "zvdx", "zvdy", "zvan", "vann", "polarization",
                                          "wavelength", "wavelengths_n", "wavelength_weights_n", "tolerance"):
                        log.info(f"Ignoring descriptor: '{descriptor.data}' with values {descriptor.children}!")
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
        for x in _:
            print(repr(x))
        surface = Surface()

        for surface_property in _:
            if isinstance(surface_property, tree.Tree):
                match surface_property.data:
                    case "surface_stop":
                        surface.stop = True
                    case "surface_name":
                        surface.name = surface_property.children[0].strip()
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
                    case "surface_glass":
                        surface.glass = surface_property.children[0].split(" ")[0]
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
    print(parse_tree.pretty())
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


def read(input_path_or_stream: Union[FileLike, PathLike]) -> OpticalSystem:
    """
    Reads a zmx file.

    :param input_path_or_stream: The file to read the optical system from, or its file-path.

    :return: A representation of the optical system.
    """
    if isinstance(input_path_or_stream, PathLike):
        input_file = open(input_path_or_stream, 'rb')
    else:
        input_file = input_path_or_stream
    with input_file:
        contents = input_file.read()
        assert len(contents) > 0, f"Empty .zmx file: 0 bytes read from {input_file.name}."
        if isinstance(contents, str):
            return __parse_zmx_str(contents)
        else:
            return __parse_zmx_bytes(contents)
