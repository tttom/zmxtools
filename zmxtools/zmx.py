import math
from typing import List, Dict, Union
from collections import defaultdict
from lark import Lark, tree, lexer, Transformer, Discard
from lark.indenter import Indenter

from zmxtools.definitions import Material, MaterialLibrary, Surface, OpticalSystem, FileLike, PathLike
from zmxtools import log

log = log.getChild(__name__)

__all__ = ["read"]


def __parse_zmx_str(contents: str) -> OpticalSystem:
    """Parses the text extracted from a .zmx file into an `OpticalSystem`."""
    # Parse into a Tree with root 'start'
    zmx_grammar = r"""
            ?start: _NL* node*

            ?node: property _NL | surface

            ?property: version | mode | name | note | unit | glass_catalog | wavelength | wavelength_weight | wavelength_with_weight
                | enpd | envd | nscd | vann
                | field_type | field_x | field_y | field_n_x | field_n_y | vignetting_cx | vignetting_cy | vignetting_dx | vignetting_dy
                | vignetting_zcx | vignetting_zcy | vignetting_zdx | vignetting_zdy
                | polarization | blank | coating_filename | unknown_property

            version: "VERS" STRING
            mode: "MODE" STRING
            name: "NAME" STRING
            note: "NOTE" INT STRING
            unit: "UNIT" CNAME STRING
            glass_catalog: "GCAT" STRING
            wavelength: "WAVL" NUMBER+
            wavelength_weight: "WWGT" NUMBER+
            wavelength_with_weight: "WAVM" NUMBER+
            field_type: "FTYP" NUMBER+
            field_x: "XFLD" NUMBER
            field_n_x: "XFLN" NUMBER+
            field_y: "YFLD" NUMBER
            field_n_y: "YFLN" NUMBER+
            vignetting_zcx: "ZVCX" NUMBER
            vignetting_cx: "VCXN" NUMBER+
            vignetting_zcy: "ZVCY" NUMBER
            vignetting_cy: "VCYN" NUMBER+
            vignetting_zdx: "ZVDX" NUMBER
            vignetting_dx: "VDXN" NUMBER+
            vignetting_zdy: "ZVDY" NUMBER
            vignetting_dy: "VDYN" NUMBER+
            enpd: "ENPD" NUMBER
            envd: "ENVD" NUMBER+
            nscd: "NSCD" NUMBER+
            vann: "VANN" NUMBER+
            blank: "BLNK"
            polarization: "POLS" NUMBER+
            coating_filename: "COFN" STRING* | "COFN QF" ESCAPED_STRING*
            unknown_property.-1: CNAME STRING*

            surface: "SURF" INT _NL _INDENT (surface_property _NL)+ _DEDENT
            ?surface_property: surface_stop | surface_name
                | surface_type | surface_curvature | surface_diameter | surface_floating_aperture
                | surface_coating | surface_mirror | surface_distance_z | surface_glass
                | surface_parameter | surface_data | unknown_surface_property

            surface_stop: "STOP"
            surface_type: "TYPE" STRING*
            surface_curvature: "CURV" NUMBER+ ESCAPED_STRING
            surface_diameter: "DIAM" NUMBER+ ESCAPED_STRING
            surface_floating_aperture: "FLAP" NUMBER+
            surface_coating: "COAT" STRING
            surface_mirror: "MIRR" NUMBER+
            surface_parameter: "PARM" INT NUMBER+
            surface_data: "XDAT" INT NUMBER+ ESCAPED_STRING
            surface_name: "COMM" STRING*
            surface_distance_z: "DISZ" NUMBER
            surface_glass: "GLAS" STRING_WITHOUT_SPACES NUMBER+
            unknown_surface_property.-1: CNAME STRING*

            _WS: /[ \t]/
            STRING: /.+/
            STRING_WITHOUT_SPACES: /[^ \t]+/
            NUMBER: SIGNED_NUMBER | "INFINITY"

            %import common.ESCAPED_STRING
            %import common.CNAME
            %import common.INT
            %import common.SIGNED_NUMBER
            %import common.WS_INLINE
            %declare _INDENT _DEDENT
            %ignore WS_INLINE

            _NL: /(\r?\n[\t ]*)+/
        """

    class TreeIndenter(Indenter):
        NL_type = '_NL'
        OPEN_PAREN_types = []
        CLOSE_PAREN_types = []
        INDENT_type = '_INDENT'
        DEDENT_type = '_DEDENT'
        tab_len = 8

    parser = Lark(zmx_grammar, parser="lalr", postlex=TreeIndenter())
    parsed_tree = parser.parse(contents)

    # Remove unknown items (strictly not needed, though these would be flagged otherwise)

    class RemoveUnknownTransformer(Transformer):
        def unknown_property(self, _):
            return Discard

        def unknown_surface_property(self, _):
            return Discard

    # parsed_tree = RemoveUnknownTransformer().transform(parsed_tree)  # To clean up the unknown items from the tree

    # Go through tree to construct an OpticalSystem with multiple Surfaces.
    optical_system = OpticalSystem()
    notes = defaultdict(str)
    surfaces = dict()
    for c in parsed_tree.children:
        match c.data:
            case "version":
                optical_system.version = c.children[0].value
            case "mode":
                mode_str = c.children[0].value
                is_sequential = mode_str.upper().startswith("SEQ")
                if not is_sequential:
                    log.warning(f"This does not seem to be a sequential model because the MODE is specified as '{mode_str}'.")
            case "name":
                optical_system.name = c.children[0].value
            case "note":
                note_idx = int(c.children[0].value)
                notes[note_idx] += c.children[1].value
            case "unit":
                unit_str = c.children[0].value.strip().upper()
                if len(unit_str) > 2:
                    unit_str = unit_str[:2]
                unit_dict = {"UM": 1e-6, "MM": 1e-3, "CM": 1e-2, "DM": 100e-3, "DA": 10.0, "HM": 100.0, "KM": 1e3, "GM": 1e9, "TM": 1e12,
                             "IN": 25.4e-3, "FT": 304.8e-3, "FE": 304.8e-3, "FO": 304.8e-3}
                optical_system.unit = unit_dict[unit_str] if unit_str in unit_dict else 1.0
            case "glass_catalog":
                optical_system.material_library = MaterialLibrary(c.children[0].value.strip())
            case "coating_filename":
                optical_system.coating_filename = ".".join(_.value.strip() for _ in c.children)
            case "wavelengths":
                optical_system.wavelengths = [float(_.value.strip()) for _ in c.children]
            case "wavelength_weights":
                optical_system.wavelength_weights = [float(_.value.strip()) for _ in c.children]
            case "surface":
                surface_idx = int(c.children[0].value)
                surface = Surface()
                surface_parameters = dict()
                surface_data = dict()
                for sc in c.children[1:]:
                    match sc.data:
                        case "surface_stop":
                            surface.stop = True
                        case "surface_type":
                            surface.type = sc.children[0].value
                        case "surface_curvature":
                            surface.curvature = float(sc.children[0].value)
                        case "surface_diameter":
                            surface.radius = float(sc.children[0].value) / 2.0
                        case "surface_coating":
                            surface.coating = sc.children[0].value
                        case "surface_mirror":
                            surface.mirror = int(sc.children[0].value) != 2
                        case "surface_parameter":
                            surface_parameters[int(sc.children[0].value)] = float(sc.children[1].value)
                        case "surface_data":
                            surface_data[int(sc.children[0].value)] = " ".join(_.value for _ in sc.children)
                        case "surface_name":
                            surface.name = " ".join(_.value for _ in sc.children)
                        case "surface_distance_z":
                            surface.distance = float(sc.children[0].value)
                        case "surface_glass":
                            surface.material = Material(sc.children[0].value.strip())
                        case s_prop if s_prop in ("surface_hide", "surface_slab", "surface_pops", "surface_flap"):
                            log.debug(f"Ignoring surface property: '{sc.data}' with values {sc.children}!")
                        case _:
                            log.warning(f"Unknown surface property '{sc}'.")
                surface.parameters = [_[1] for _ in sorted(surface_parameters.items(), key=lambda _: _[0])]
                surface.data = [_[1] for _ in sorted(surface_data.items(), key=lambda _: _[0])]
                surfaces[surface_idx] = surface
            case "wavelength":
                optical_system.wavelengths = [float(_.value) * 1e-6 for _ in c.children]  # TODO: Always in micrometer?
            case "wavelength_weight":
                optical_system.wavelength_weights = [float(_.value) for _ in c.children]
            case "coating_filename":
                optical_system.coating_filename = c.children[0].value
            case desc if desc in ("enpd", "envd", "gfac", "ray_aim", "push", "sdma", "ftyp", "ropd", "picb",
                                      "pwav", "glrs", "gstd", "nscd", "fwgn", "mnum", "moff",
                                      "xfld", "yfld", "xfln", "yfln", "vcxn", "vcyn", "vdxn", "vdyn",
                                      "fwgt", "zvcx", "zvcy", "zvdx", "zvdy", "zvan", "vann", "polarization",
                                      "wavelength", "wavelengths_n", "wavelength_weights_n", "tolerance", "blank", ):
                log.info(f"Ignoring descriptor: '{c.data}' with values {c.children}!")
            case _:
                log.warning(f"Unknown descriptor: '{c.data}' with values {c.children}!")

    # Put everything together
    # Sort the dictionaries by key and add to the OpticalSystem
    optical_system.description = "\n".join(_[1] for _ in sorted(notes.items(), key=lambda _: _[0]))
    optical_system.surfaces = [_[1].using_unit(optical_system.unit) for _ in sorted(surfaces.items(), key=lambda _: _[0])]

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
