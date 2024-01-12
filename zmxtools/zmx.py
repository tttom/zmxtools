from __future__ import annotations

from collections.abc import Iterator
import numpy as np
import pathlib
import re
from typing import Tuple, Optional, Sequence

from zmxtools.utils.io import FileLike, PathLike
from zmxtools.utils.array import array_like, array_type, asarray, norm
from zmxtools.utils import zernike
from zmxtools.optical_design.source import Source
from zmxtools.optical_design.light import Wavefront
from zmxtools.optical_design.material import Material, VACUUM, MaterialLibrary
from zmxtools.optical_design.medium import Medium, HomogeneousMedium
from zmxtools.optical_design.optic import CompoundElement, SurfaceDetector, OpticalDesign
from zmxtools.optical_design.geometry import Translation, EulerRotation, SphericalTransform
from zmxtools.optical_design.surface import Surface, DiskAperture, SnellInterface
from zmxtools.agf import AgfMaterialLibrary
from zmxtools.parser import OrderedCommandDict, Command
from zmxtools import log

log = log.getChild(__name__)

__all__ = ["ZmxOrderedCommandDict", "ZmxOpticalDesign", "ZmxSurface"]


class ZmxOrderedCommandDict(OrderedCommandDict):
    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 2) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text extracted from a zmx file.

        :param file_contents: The text string extracted from an optical file.
        :param spaces_per_indent: The number of spaces per indent to assumed

        :return: The command dictionary.
        """
        def parse_section(lines: Iterator[str],
                          parent_indent: int = -1, section_indent: int = 0,
                          out: Optional[OrderedCommandDict] = None) -> Tuple[OrderedCommandDict, Optional[Command]]:
            """
            Auxiliary recursive function to parse a section of the file's lines.

            :param lines: The iterable with the lines to parse.
            :param parent_indent: The indent of the enclosing section.
            :param section_indent: The indent of the current section.
            :param out: The optional collection to add to as the result.

            :return: A collection of commands, corresponding to the lines in this section.
            """
            if out is None:
                out = OrderedCommandDict(spaces_per_indent=spaces_per_indent)
            try:
                while (line := next(lines)) is not None:
                    match = re.match(r"(\s*)(\S+)(\s.*)?", line)
                    if match is None:
                        continue  # skip empty line
                    indent_str, command_name, command_argument = match.groups()
                    if command_argument is not None:
                        command_argument = command_argument[1:]
                    if '\t' not in indent_str:
                        indent = len(indent_str)
                    else:  # Replace tabs with spaces before counting indentation
                        indent = sum((spaces_per_indent - (_ % spaces_per_indent)) if c == '\t' else 1 for _, c in enumerate(indent_str))
                    next_command = Command(name=command_name, argument=command_argument)
                    if indent <= parent_indent:
                        return out, next_command  # pass next command to
                    elif parent_indent < indent <= section_indent:  # be very forgiving
                        out.append(next_command)
                    else:  # indent > section_indent:  recurse
                        out[-1].children, next_command = parse_section(
                            lines,
                            parent_indent=section_indent, section_indent=indent,
                            out=OrderedCommandDict([next_command], spaces_per_indent=spaces_per_indent))
                        if next_command is not None:
                            out.append(next_command)
            except StopIteration:
                pass
            return out, None

        return parse_section(file_contents.splitlines().__iter__())[0]


class ZmxSource(Source):
    def __init__(self, medium: Medium,
                 E: array_like, H: array_like,
                 p: array_like, d: array_like,
                 wavelengths: array_like, wavelength_weights: array_like,
                 surface: ZmxSurface):
        self.wavelengths = asarray(wavelengths)
        self.wavelength_weights = asarray(wavelength_weights)
        self.surface: ZmxSurface = surface

        p = asarray(p)
        E = asarray(E)
        H = asarray(H)
        k0 = 2.0 * np.pi / self.wavelengths
        k = d / norm(d) * k0 * medium.complex_refractive_index(wavenumber=k0, p=p, E=E, H=H)
        super().__init__(medium=medium, wavefront=Wavefront(E=E, H=H, p=p, k=k, d=d, k0=k0))


class ZmxOpticalDesign(OpticalDesign):
    @staticmethod
    def from_str(contents: str, spaces_per_indent: int = 2) -> ZmxOpticalDesign:
        """Parses the text extracted from a .zmx file into an `OpticalDesign`."""
        return ZmxOpticalDesign(ZmxOrderedCommandDict.from_str(contents, spaces_per_indent=spaces_per_indent))

    @staticmethod
    def from_file(input_path_or_stream: FileLike | PathLike,
                  material_libraries: Sequence[PathLike | MaterialLibrary] = tuple[PathLike | MaterialLibrary](),
                  spaces_per_indent: int = 2,
                  encoding: str = 'utf-16') -> ZmxOpticalDesign:
        """
        Reads a zmx file into an `OpticalDesign` representation.

        :param input_path_or_stream: The file to read the optical system from, or its file-path.
        :param material_libraries: List of MaterialLibraries or paths to AGF files that can be used as glass catalogs.
        :param spaces_per_indent: The optional number of spaces per indent/tab.
        :param encoding: The text-encoding to try first.

        :return: A representation of the optical system.
        """
        return ZmxOpticalDesign(ZmxOrderedCommandDict.from_file(input_path_or_stream,
                                                                spaces_per_indent=spaces_per_indent,
                                                                encoding=encoding),
                                material_libraries)

    def __init__(self, commands: OrderedCommandDict,
                 material_libraries: Sequence[PathLike | MaterialLibrary] = tuple[PathLike | MaterialLibrary]()):
        """
        Constructs an OpticalDesign from a parsed command dictionary.

        :param commands: The command dictionary obtained from parsing a ZMX file.
        :param material_libraries: List of MaterialLibraries or paths to AGF files that can be used as glass catalogs.
        """
        self.commands = commands

        self.version = self.commands["VERS", 0].argument if "VERS" in self.commands else ""
        log.debug(f'Loading a zmx file with version "{self.version}"...')
        self.sequential = True
        if "MODE" in self.commands:
            mode = self.commands["MODE", 0].words[0]
            if mode != "SEQ":
                if mode == "NSC":
                    log.warning(f"Non-sequential mode not implemented.")
                else:
                    log.warning(f"Unrecognized mode {mode}.")
                self.sequential = False
        self.name = self.commands["NAME", 0].argument if "NAME" in self.commands else ""
        self.author = self.commands["AUTH", 0].argument if "AUTH" in self.commands else ""
        self.description = '\n'.join(_.argument.replace('\n', '') for _ in self.commands.sort_and_merge("NOTE")) \
            if "NOTE" in self.commands else ""
        self.unit: float = 1.0
        if "UNIT" in self.commands:
            unit_code = self.commands["UNIT", 0].argument.split(maxsplit=1)[0]
            unit_dict = {"UM": 1e-6, "MM": 1e-3, "CM": 1e-2, "DM": 100e-3, "METER": 1.0, "ME": 1.0, "M": 1.0,
                         "DA": 10.0, "HM": 100.0, "KM": 1e3, "GM": 1e9, "TM": 1e12,
                         "IN": 25.4e-3, "FEET": 304.8e-3, "FT": 304.8e-3, "FE": 304.8e-3, "FO": 304.8e-3}
            if unit_code in unit_dict:
                self.unit = unit_dict[unit_code]
            else:
                log.warning(f"Unrecognized unit code {unit_code}. Defaulting to 1m.")
        log.info(f'Loading optical design "{self.name}" by "{self.author}": {self.description} using units of {self.unit}.')

        log.debug("Configuring material libraries...")
        self.material_libraries = [_ for _ in material_libraries if isinstance(_, MaterialLibrary)]
        material_library_file_paths = [_ for _ in material_libraries if not isinstance(_, MaterialLibrary)]
        if "GCAT" in self.commands:
            for name in self.commands["GCAT", 0].words:
                if name not in (_.name for _ in self.material_libraries):
                    material_library = None
                    for material_library_file_path in material_library_file_paths:
                        if not isinstance(material_library_file_path, pathlib.Path):
                            material_library_file_path = pathlib.Path(material_library_file_path)
                        if name == material_library_file_path.stem.upper():
                            material_library = AgfMaterialLibrary.from_file(material_library_file_path)
                            break
                    if material_library is None:
                        log.warning(f"Glass catalog {name} not found in {material_library_file_paths}.")
                    else:
                        self.material_libraries.append(material_library)

        log.debug("Configuring coatings...")
        self.coating_filenames = list[str]()
        # Coating
        if "COFN" in self.commands:
            file_names = self.commands["COFN", 0].words
            if file_names[0] == "QF":
                file_names = file_names[1:]
            self.coating_filenames += file_names
        log.info(f"Coating files {self.coating_filenames}. Coatings are not yet implemented.")

        self.background_material = VACUUM  # CiddorAir()
        surfaces: Sequence[ZmxSurface] = [ZmxSurface(s.children, unit=self.unit,
                                                     material_libraries=self.material_libraries,   # todo: wavelengths are specified relative to the refractive index in air at 20+273.15K and 101.325Pa!
                                                     background_material=self.background_material)
                                          for s in self.commands.sort_and_merge("SURF")]
        log.info(f"Detected {len(surfaces)} surfaces, including the object and image surface.")

        media = [HomogeneousMedium(_.material) for _ in surfaces[:-1]]
        object_medium = media[0]

        log.debug("Reading the illumination spectrum...")
        wavelengths = self.commands["WAVL", 0].numbers if "WAVL" in self.commands else list[float]()  # "WAVM" doesn't seem very reliable. Perhaps this depends on the version?
        wavelength_weights = self.commands["WWGT", 0].numbers if "WWGT" in self.commands else list[float]()
        if len(wavelength_weights) < len(wavelengths):
            wavelength_weights = [*wavelength_weights, *([1.0] * (len(wavelengths) - len(wavelength_weights)))]
        if len(wavelengths) == 0 and "WAVM" in self.commands:  # This seems to be the new way, but it contains many unused wavelengths as well
            wavelengths_and_weights = [_.numbers[:2] for _ in self.commands.sort_and_merge("WAVM")]
            unique_wavelengths = set(_[0] for _ in wavelengths_and_weights)
            nb_occurences = [sum(u == _[0] for _ in wavelengths_and_weights) for u in unique_wavelengths]
            unique_wavelengths = [_ for _, n in zip(unique_wavelengths, nb_occurences) if n == 1]
            wavelengths_and_weights = [_ for _ in wavelengths_and_weights if _[0] in unique_wavelengths]

            wavelengths = [_[0] for _ in wavelengths_and_weights]
            wavelength_weights = [_[1] for _ in wavelengths_and_weights]

        # Make all units meters
        wavelengths = [_ * 1e-6 for _ in wavelengths]
        log.info(f"Using wavelengths of {[f'{_ / 1e-9:0.1f}' for _ in wavelengths]} nm.")

        log.debug("Parsing the field configuration...")
        self.field_comment = self.commands["FCOM", 0].argument if "FCOM" in self.commands else ""
        self.numerical_aperture = 1.0
        if "FNUM" in self.commands:
            f_number = self.commands["FNUM", 0].numbers[0]
            self.numerical_aperture_image = 2.0 / f_number  # todo: account for refractive index of object or image space?
        if "OBNA" in self.commands:
            self.numerical_aperture_object = self.commands["OBNA", 0].numbers[0]
        if "ENPD" in self.commands:
            pupil_radius_object = self.commands["ENPD", 0].numbers[0] / 2.0
        if "EFFL" in self.commands:
            effective_focal_length = self.commands["EFFL", 0].numbers[0]
        if "FTYP" in self.commands:
            field_type = self.commands["FTYP", 0].numbers[0]  # []
            field_as_height = (field_type % 2) == 1   # angle: False, height: True
            field_at_image = (field_type // 2) == 1  # object: False, image: True
        # field also uses VDXN, VDYN, VCXN, VXYN, VANN, VWGN, VWGT

        source = ZmxSource(medium=object_medium,
                           E=(1, 0, 0), H=(0, 1, 0),
                           p=0, d=(0, 0, 1),
                           wavelengths=wavelengths, wavelength_weights=wavelength_weights,
                           surface=surfaces[0])
        optic = CompoundElement(*surfaces[1:-1], *media[1:-1])
        image_medium = media[-1]
        detector = SurfaceDetector(medium=image_medium, surface=surfaces[-1])

        super().__init__(source=source, optic=optic, detector=detector)
        self.source: ZmxSource = source  # A more specific type
        self.detector: SurfaceDetector = detector  # A more specific type

    @property
    def surfaces(self) -> Sequence[ZmxSurface]:
        """All ZmxSurfaces in order, including the object and the image surface."""

        def all_surfaces(optic) -> Sequence[ZmxSurface]:
            result = list[ZmxSurface]()
            if isinstance(optic, CompoundElement):
                for e in optic.elements:
                    result += all_surfaces(e)
            elif isinstance(optic, ZmxSurface):
                result.append(optic)
            return result

        return self.source.surface, *all_surfaces(self.optic), self.detector.surface

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"


class ZmxSurface(Surface):
    """A class to represent a thin surface between two volumes as read from a .zmx file."""
    def __init__(self, commands: OrderedCommandDict, unit: float = 1.0,
                 material_libraries: Sequence[MaterialLibrary] = tuple[MaterialLibrary](),
                 background_material: Material = VACUUM):
        """
        Construct a new surface based from a command dictionary that represents the corresponding lines in the file.

        :param commands: The command dictionary.
        :param unit: The unit (in meters) of the lengths in the command dictionary.
        :param material_libraries: A collection of material libraries from which to choose glasses.
        :param background_material: The material to use when no (recognized) material is specified.
        """
        self.unit = unit
        self.commands: OrderedCommandDict = commands

        self.type = self.commands["TYPE", 0].argument if "TYPE" in self.commands else "STANDARD"
        # Types: "STANDARD" , "EVENASPH", "TOROIDAL", "XOSPHERE", "COORDBRK", "TILTSURF", "PARAXIAL", "DGRATING"
        self.curvature = self.commands["CURV", 0].numbers[0] / self.unit if "CURV" in self.commands else 0.0
        self.coating = self.commands["COAT", 0].words[0] if "COAT" in self.commands and len(self.commands["COAT", 0].words) > 0 else ""
        self.radius = self.commands["DIAM", 0].numbers[0] * self.unit / 2.0 if "DIAM" in self.commands else np.inf
        self.stop = "STOP" in self.commands
        self.distance = self.commands["DISZ", 0].numbers[0] * self.unit if "DISZ" in self.commands else np.inf
        self.comment = self.commands["COMM", 0].argument if "COMM" in self.commands else ""
        glass_name = self.commands["GLAS", 0].words[0] if "GLAS" in self.commands and len(self.commands["GLAS", 0].words) else ""
        self.reflect = glass_name == "MIRROR"  # Not "MIRR" command for some reason
        if glass_name == "" or glass_name == "MIRROR":
            material = background_material
        else:
            material = None
            for material_library in material_libraries:
                if glass_name in material_library:
                    material = material_library.find_all(glass_name)[0]
                    break
            if material is None:
                log.error(f"Glass {glass_name} not found in {material_libraries}.")
                material = Material(name=glass_name)  # Dummy material
        self.material: Material = material
        # self.floating_aperture = self.commands["FLAP", 0].numbers if "FLAP" in self.commands else 0.0
        self.conic_constant = self.commands["CONI", 0].numbers if "CONI" in self.commands else 0.0
        self.parameters = [_.numbers[0] for _ in self.commands.sort_and_merge("PARM")]
        self.extra_data = self.commands["XDAT", 0].numbers if "XDAT" in self.commands else []
        aperture_offsets = self.commands["OBDC", 0].numbers[:2] if "OBDC" in self.commands else []
        pickup_parameter_commands = self.commands["PPAR"]  # of the form PPAR parameter from_surface factor offset 0
        for pickup_parameter_command in pickup_parameter_commands:
            parameter_index, from_surface, factor, offset = pickup_parameter_command.numbers[:4]
            parameter_index -= 1
            # self.parameter[parameter_index] = surface[from_surface].parameters[parameter_index] * factor + offset
            # The number on file seem to be computed already.

        def standard_sag(r2: array_like) -> array_type:
            return self.curvature * r2 / (1 + (1 - (1 + self.conic_constant) * self.curvature ** 2 * r2)**0.5)

        def odd_asphere_sag(r2: array_like, coefficients: array_like) -> array_type:
            result = 0.0
            r = r2 ** 0.5
            for _, c in enumerate(coefficients):
                result = result + c * (r ** (_ + 1))
            return standard_sag(r2) * result

        def even_asphere_sag(r2: array_like, coefficients: array_like) -> array_type:
            result = 0.0
            for _, c in enumerate(coefficients):
                result = result + c * (r2 ** (_ + 1))
            return standard_sag(r2) * result

        def zernike_sag(position: array_like, coefficients: array_like, indices: array_like = tuple(),
                        radius: array_type = 1.0) -> array_type:
            position = asarray(position)
            rho = norm(position[..., :2]) / radius
            phi = np.arctan2(position[..., 1], position[..., 0])
            z = zernike.Polynomial(coefficients=coefficients, indices=indices)
            return z(rho, phi)

        def poly_sag(position: array_type, coefficients: array_like, radius: array_type = 1.0) -> array_type:
            p = asarray(position) / radius
            result = 0.0
            for _, c in enumerate(coefficients):
                if c != 0.0:
                    # 0: x^1 y^0, 1: x^0 y^1,    2: x^2 y^0, 3: x^1 y^1, 4: x^0 y^2,  5: x^3 y^0, ...
                    # j = n * (n+1) / 2 - 1 -> n * (n+1) / 2 - 1 + n
                    nb_factors = np.floor((2 * (_ + 1)) ** 0.5).astype(int)
                    exponent_y = _ - nb_factors * (nb_factors + 1) // 2 + 1
                    exponents = [nb_factors - exponent_y, exponent_y]
                    result = result + c * (p[..., 0] ** exponents[0]) * (p[..., 1] ** exponents[1])
            return result

        match self.type:
            case "COORDBRK":  # A Transform, not a Surface
                self.decenter_xy = asarray(self.parameters[:2]) * self.unit
                self.euler_angles = [_ * np.pi / 180.0 for _ in self.parameters[2:5]]  # Rotation of the coordinate system of the optics. Euler angles are applied in the order x, y, z.
                self.rotate_before_decenter = self.parameters[5] != 0  # If False, decenter, then apply Euler angles; and then translate thickness. If True, apply Euler angles; then decenter and thickness.
            case "PARAXIAL":
                self.focal_length = self.parameters[0] * self.unit
                opd_calc_mode = self.parameters[1]
            case "STANDARD":  # STANDARD surface: z**2 == 2 * r * self.curvature - (1 + self.conic_constant) * r**2, used as the basis for many other surfaces
                self.sag = lambda position: standard_sag(np.sum(asarray(position)[..., :2] ** 2))
            case "EVENASPH":  # Even Asphere Surface, used as the basis for many other surfaces
                self.sag = lambda position: even_asphere_sag(np.sum(asarray(position)[..., :2] ** 2),
                                                             asarray(self.parameters) * self.unit)  # In lens units, the extended version uses normalized radii, rho
            case "TOROIDAL":
                # self.parameters [extrapolate_zernike, radius_of_rotation, coefficients]
                # data_x [nb_zernikes, norm_radius, *zernike_terms], and VPAR also seems to contain some info?
                def toroidal_sag(position: array_type) -> array_like:
                    position = asarray(position)
                    z_in_plane = even_asphere_sag(position[..., 1] ** 2, self.parameters[2:])
                    radius_of_rotation = self.parameters[1]  # if self.parameters[1] != 0.0 else np.inf
                    curved_in_x = radius_of_rotation != 0.0
                    z_to_origin_2 = (z_in_plane - radius_of_rotation) ** 2 - position[..., 1] ** 2
                    toroid = (z_to_origin_2 ** 0.5 + radius_of_rotation) * curved_in_x + (1 - curved_in_x) * z_in_plane
                    return toroid + zernike_sag(position,
                                                coefficients=asarray(self.extra_data[2:]) * self.unit,
                                                radius=self.extra_data[1] * self.unit)
                self.sag = toroidal_sag
            case "SZERNSAG":  # Zernike Standard Sag Surface is derived from the Even aspheric surface
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * self.unit
                self.sag = lambda position: even_asphere_sag(np.sum(asarray(position)[..., :2] ** 2),
                                                             asarray(self.parameters) * self.unit) + \
                                            zernike_sag(position, self.zernike_coefficients, radius=self.zernike_radius)
            case "SZERNPHA":  # STANDARD sag, but opd changed by Zernikes
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * 2 * np.pi
                self.sag = lambda position: even_asphere_sag(np.sum(asarray(position)[..., :2] ** 2),
                                                             asarray(self.parameters) * self.unit)
                self.phase = lambda position: zernike_sag(position, self.zernike_coefficients, radius=self.zernike_radius)
            case "FZERNSAG":  # Zernike Fringe Sag Surface is derived from the Even aspheric surface
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * self.unit
                self.sag = lambda position: even_asphere_sag(np.sum(asarray(position)[..., :2] ** 2),
                                                             asarray(self.parameters) * self.unit) + \
                                            zernike_sag(
                                                position, self.zernike_coefficients,
                                                indices=zernike.fringe2index(range(1, 1 + len(self.zernike_coefficients))),
                                                radius=self.zernike_radius)
            case "FZERNPHA":  # STANDARD sag, but opd changed by Zernikes
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * 2 * np.pi
                self.sag = lambda position: even_asphere_sag(np.sum(asarray(position)[..., :2] ** 2),
                                                             asarray(self.parameters) * self.unit)
                self.phase = lambda position: zernike_sag(
                    position, self.zernike_coefficients,
                    indices=zernike.fringe2index(range(1, 1 + len(self.zernike_coefficients))),
                    radius=self.zernike_radius)
            case "XPOLYNOM":
                self.sag = lambda position: standard_sag(np.sum(asarray(position)[..., :2] ** 2)) + \
                                            poly_sag(position, self.extra_data[2:],
                                                     radius=self.extra_data[1] * self.unit)
            case "TILTSURF":
                # A planar surface with a rotation before and its reverse after.
                self.euler_angles = np.arctan(self.parameters[:2])  # TODO convert to a Transform before and a Transform after
                self.sag = lambda position: position[..., 0] * 0.0
            case "GRID_SAG":
                log.warning(f"Surface {self.type}({self.parameters}, {self.extra_data}) not implemented:\n{self.commands}")
            case "BICONICX":
                log.warning(f"Surface {self.type}({self.parameters}, {self.extra_data}) not implemented:\n{self.commands}")
            case "IRREGULA":
                log.warning(f"Surface {self.type}({self.parameters}, {self.extra_data}) not implemented:\n{self.commands}")
            case _:
                log.warning(f"Unknown surface type: '{self.type}'!")

        super().__init__(interface=SnellInterface(transform=SphericalTransform(curvature=self.curvature)),
                         aperture=DiskAperture(outer_radius=self.radius),
                         transform=Translation([0, 0, self.distance]))

    @property
    def eccentricity(self):
        return (-self.conic_constant)**0.5

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.
        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(commands={repr(self.commands)})"
