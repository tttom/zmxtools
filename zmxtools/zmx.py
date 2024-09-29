from __future__ import annotations

import numpy as np
import pathlib
import re
from collections.abc import Iterator
from typing import Optional, Sequence, Tuple

from zmxtools import log

from zmxtools.agf import AgfMaterialLibrary
from zmxtools.optical_design.geometry import SphericalTransform, Translation
from zmxtools.optical_design.light import Wavefront
from zmxtools.optical_design.material import MaterialLibrary, Material, VACUUM
from zmxtools.optical_design.medium import HomogeneousMedium, Medium
from zmxtools.optical_design.optic import CompoundElement, OpticalDesign, SurfaceDetector
from zmxtools.optical_design.source import Source
from zmxtools.optical_design.surface import DiskAperture, SnellInterface, Surface
from zmxtools.parser import Command, OrderedCommandDict
from zmxtools.utils import zernike
from zmxtools.utils.array import array_like, array_type, asarray, norm
from zmxtools.utils.io import FileLike, PathLike

log = log.getChild(__name__)

__all__ = ['ZmxOpticalDesign', 'ZmxSource', 'ZmxSurface']


class ZmxOrderedCommandDict(OrderedCommandDict):
    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 2) -> OrderedCommandDict:
        """
        Create a new `OrderedCommandDict` from a multi-line text extracted from a zmx file.

        :param file_contents: The text string extracted from an optical file.
        :param spaces_per_indent: The number of spaces per indent to assumed

        :return: The command dictionary.
        """
        def parse_section(
            lines: Iterator[str],
            parent_indent: int = -1,
            section_indent: int = 0,
            out: Optional[OrderedCommandDict] = None,
        ) -> Tuple[OrderedCommandDict, Optional[Command]]:
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
                    match = re.match(r'(\s*)(\S+)(\s.*)?', line)
                    if match is None:
                        continue  # skip empty line
                    indent_str, command_name, command_argument = match.groups()
                    if command_argument is not None:
                        command_argument = command_argument[1:]
                    if '\t' not in indent_str:
                        indent = len(indent_str)
                    else:  # Replace tabs with spaces before counting indentation
                        indent = (sum((spaces_per_indent - (_ % spaces_per_indent))
                                      if c == '\t' else 1 for _, c in enumerate(indent_str)
                                      )
                                  )
                    next_command = Command(name=command_name, argument=command_argument)
                    if indent <= parent_indent:
                        return out, next_command  # pass next command to
                    elif parent_indent < indent <= section_indent:  # be very forgiving
                        out.append(next_command)
                    else:  # indent > section_indent:  recurse
                        out[-1].children, next_command = parse_section(
                            lines,
                            parent_indent=section_indent,
                            section_indent=indent,
                            out=OrderedCommandDict([next_command], spaces_per_indent=spaces_per_indent),
                        )
                        if next_command is not None:
                            out.append(next_command)
            except StopIteration:
                pass
            return out, None

        return parse_section(iter(file_contents.splitlines()))[0]


class ZmxSource(Source):
    """
    A class to represent how the light source is modeled.

    It holds the starting rays and wavelengths.
    """
    def __init__(self, medium: Medium,
                 E: array_like, H: array_like,
                 p: array_like, d: array_like,
                 wavelengths: array_like, wavelength_weights: array_like,
                 surface: ZmxSurface,
                 ):
        """
        Construct a new light-source from a collection of ray starting points.

        All arguments are broadcast to a common array shape.

        :param medium: The medium that the light-source is immersed in.
        :param E: The electric field vector of each ray.
        :param H: The magnetic field vector of each ray.
        :param p: The initial position of each ray.
        :param d: The direction of each ray.
        :param wavelengths: The wavelengths to represent the spectrum.
        :param wavelength_weights: The intensity weights for each wavelength.
        :param surface: The initial surface.
        """
        self.wavelengths = asarray(wavelengths).real
        self.wavelength_weights = asarray(wavelength_weights).real
        self.surface: ZmxSurface = surface

        electric_field = asarray(E)
        magnetizing_field = asarray(H)
        position = asarray(p)
        direction = asarray(d)

        k0 = 2.0 * np.pi / self.wavelengths
        k = d / norm(d) * k0 * medium.complex_refractive_index(
            wavenumber=k0, p=position, E=electric_field, H=magnetizing_field,
        )
        super().__init__(medium=medium, wavefront=Wavefront(
            E=electric_field, H=magnetizing_field, p=position, k=k, d=direction, k0=k0,
        ))


class ZmxOpticalDesign(OpticalDesign):
    """A class to represent the complete optical design."""
    @staticmethod
    def from_str(file_contents: str, spaces_per_indent: int = 2) -> ZmxOpticalDesign:
        """Parses the text extracted from a .zmx file into an `OpticalDesign`."""
        return ZmxOpticalDesign(ZmxOrderedCommandDict.from_str(file_contents, spaces_per_indent=spaces_per_indent))

    @staticmethod
    def from_file(input_path_or_stream: FileLike | PathLike,
                  material_libraries: Sequence[PathLike | MaterialLibrary] = tuple[PathLike | MaterialLibrary](),
                  spaces_per_indent: int = 2,
                  encoding: str = 'utf-16',
                  ) -> ZmxOpticalDesign:
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
                                material_libraries,
                                )

    def __init__(self, commands: OrderedCommandDict,
                 material_libraries: Sequence[PathLike | MaterialLibrary] = tuple[PathLike | MaterialLibrary]()):
        """
        Constructs an OpticalDesign from a parsed command dictionary.

        :param commands: The command dictionary obtained from parsing a ZMX file.
        :param material_libraries: List of MaterialLibraries or paths to AGF files that can be used as glass catalogs.
        """
        self.commands = commands

        self.version = self.commands['VERS', 0].argument if 'VERS' in self.commands else ''
        log.debug(f'Loading a zmx file with version "{self.version}"...')
        self.sequential = True
        if 'MODE' in self.commands:
            mode = self.commands['MODE', 0].words[0]
            if mode != 'SEQ':
                if mode == 'NSC':
                    log.warning('Non-sequential mode not implemented.')
                else:
                    log.warning(f'Unrecognized mode {mode}.')
                self.sequential = False
        self.name = self.commands['NAME', 0].argument if 'NAME' in self.commands else ''
        self.author = self.commands['AUTH', 0].argument if 'AUTH' in self.commands else ''
        self.description = ('\n'.join(_.argument.replace('\n', '') for _ in self.commands.sort_and_merge('NOTE'))
                            if 'NOTE' in self.commands else ''
                            )
        self.unit: float = 1.0
        if 'UNIT' in self.commands:
            unit_code = self.commands['UNIT', 0].argument.split(maxsplit=1)[0]
            unit_dict = {'UM': 1e-6, 'MM': 1e-3, 'CM': 1e-2, 'DM': 100e-3, 'METER': 1.0, 'ME': 1.0, 'M': 1.0,
                         'DA': 10.0, 'HM': 100.0, 'KM': 1e3, 'GM': 1e9, 'TM': 1e12,
                         'IN': 25.4e-3, 'FEET': 304.8e-3, 'FT': 304.8e-3, 'FE': 304.8e-3, 'FO': 304.8e-3,
                         }
            if unit_code in unit_dict:
                self.unit = unit_dict[unit_code]
            else:
                log.warning(f'Unrecognized unit code {unit_code}. Defaulting to 1m.')
        log.info(f'Loading optical design "{self.name}" by "{self.author}": {self.description}, units of {self.unit}.')

        log.debug('Configuring material libraries...')
        self.material_libraries = [_ for _ in material_libraries if isinstance(_, MaterialLibrary)]
        material_library_file_paths = [_ for _ in material_libraries if not isinstance(_, MaterialLibrary)]
        if 'GCAT' in self.commands:
            for name in self.commands['GCAT', 0].words:
                if all(_.name != name for _ in self.material_libraries):
                    material_library = None
                    for material_library_file_path in material_library_file_paths:
                        if not isinstance(material_library_file_path, pathlib.Path):
                            material_library_file_path = pathlib.Path(material_library_file_path)
                        if name == material_library_file_path.stem.upper():
                            material_library = AgfMaterialLibrary.from_file(material_library_file_path)
                            break
                    if material_library is None:
                        log.warning(f'Glass catalog {name} not found in {material_library_file_paths}.')
                    else:
                        self.material_libraries.append(material_library)

        log.debug('Configuring coatings...')
        self.coating_filenames = list[str]()
        # Coating
        if 'COFN' in self.commands:
            file_names = self.commands['COFN', 0].words
            if file_names[0] == 'QF':
                file_names = file_names[1:]
            self.coating_filenames += file_names
        log.info(f'Coating files {self.coating_filenames}. Coatings are not yet implemented.')

        self.background_material = VACUUM  # CiddorAir()
        # todo: wavelengths are specified relative to the refractive index in air at 20+273.15K and 101.325Pa!
        surfaces: Sequence[ZmxSurface] = [ZmxSurface(s.children, unit=self.unit,
                                                     material_libraries=self.material_libraries,
                                                     background_material=self.background_material,
                                                     )
                                          for s in self.commands.sort_and_merge('SURF')]
        log.info(f'Detected {len(surfaces)} surfaces, including the object and image surface.')

        media = [HomogeneousMedium(_.material) for _ in surfaces[:-1]]
        object_medium = media[0]

        log.debug('Reading the illumination spectrum...')
        # Note that 'WAVM' doesn't seem very reliable. Perhaps this depends on the version?
        wavelengths = self.commands['WAVL', 0].numbers if 'WAVL' in self.commands else list[float]()
        wavelength_weights = self.commands['WWGT', 0].numbers if 'WWGT' in self.commands else list[float]()
        if len(wavelength_weights) < len(wavelengths):
            wavelength_weights = [*wavelength_weights, *([1.0] * (len(wavelengths) - len(wavelength_weights)))]
        if len(wavelengths) == 0 and 'WAVM' in self.commands:
            # This seems to be the new way, but it contains many unused wavelengths as well
            wavelengths_and_weights = [_.numbers[:2] for _ in self.commands.sort_and_merge('WAVM')]
            unique_wavelengths = {_[0] for _ in wavelengths_and_weights}
            nb_occurences = [sum(u == _[0] for _ in wavelengths_and_weights) for u in unique_wavelengths]
            unique_wavelengths = [_ for _, n in zip(unique_wavelengths, nb_occurences) if n == 1]
            wavelengths_and_weights = [_ for _ in wavelengths_and_weights if _[0] in unique_wavelengths]

            wavelengths = [_[0] for _ in wavelengths_and_weights]
            wavelength_weights = [_[1] for _ in wavelengths_and_weights]

        # Make all units meters
        wavelengths = [_ * 1e-6 for _ in wavelengths]
        log.info(f"Using wavelengths of {[f'{_ / 1e-9:0.1f}' for _ in wavelengths]} nm.")

        log.debug('Parsing the field configuration...')
        self.field_comment = self.commands['FCOM', 0].argument if 'FCOM' in self.commands else ''
        self.numerical_aperture = 1.0
        if 'FNUM' in self.commands:
            f_number = self.commands['FNUM', 0].numbers[0]
            self.numerical_aperture_image = 2.0 / f_number  # todo: account for refractive index of object/image space?
        if 'OBNA' in self.commands:
            self.numerical_aperture_object = self.commands['OBNA', 0].numbers[0]
        if 'ENPD' in self.commands:
            pupil_radius_object = self.commands['ENPD', 0].numbers[0] / 2.0
        if 'EFFL' in self.commands:
            effective_focal_length = self.commands['EFFL', 0].numbers[0]
        if 'FTYP' in self.commands:
            field_type = self.commands['FTYP', 0].numbers[0]  # []
            field_as_height = (field_type % 2) == 1   # angle: False, height: True
            field_at_image = (field_type // 2) == 1  # object: False, image: True
        # field also uses VDXN, VDYN, VCXN, VXYN, VANN, VWGN, VWGT

        source = ZmxSource(
            medium=object_medium,
            E=(1, 0, 0), H=(0, 1, 0),
            p=0, d=(0, 0, 1),
            wavelengths=asarray(wavelengths)[:, np.newaxis],
            wavelength_weights=asarray(wavelength_weights)[:, np.newaxis],
            surface=surfaces[0],
        )
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
            surface_list = list[ZmxSurface]()
            if isinstance(optic, CompoundElement):
                surface_list = sum((all_surfaces(e) for e in optic.elements), start=surface_list)
            elif isinstance(optic, ZmxSurface):
                surface_list.append(optic)
            return surface_list

        return self.source.surface, *all_surfaces(self.optic), self.detector.surface

    def __str__(self) -> str:
        """
        Return the text string from which this object is parsed.

        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        """The representation as a constructor with all the commands as an argument."""
        return f'{self.__class__.__name__}(commands={repr(self.commands)})'


class ZmxSurface(Surface):
    """
    A class to represent a thin surface between two volumes as read from a `zmx` file.
    """
    def __init__(self, commands: OrderedCommandDict, unit: float = 1.0,
                 material_libraries: Sequence[MaterialLibrary] = tuple[MaterialLibrary](),
                 background_material: Material = VACUUM,
                 ):
        """
        Construct a new surface based from a command dictionary that represents the corresponding lines in the file.

        :param commands: The command dictionary.
        :param unit: The unit (in meters) of the lengths in the command dictionary.
        :param material_libraries: A collection of material libraries from which to choose glasses.
        :param background_material: The material to use when no (recognized) material is specified.
        """
        self.unit = unit
        self.commands: OrderedCommandDict = commands

        self.type = self.commands['TYPE', 0].argument if 'TYPE' in self.commands else 'STANDARD'
        # Types: 'STANDARD' , 'EVENASPH', 'TOROIDAL', 'XOSPHERE', 'COORDBRK', 'TILTSURF', 'PARAXIAL', 'DGRATING'
        self.curvature = self.commands['CURV', 0].numbers[0] / self.unit if 'CURV' in self.commands else 0
        self.coating = (self.commands['COAT', 0].words[0]
                        if 'COAT' in self.commands and len(self.commands['COAT', 0].words) > 0 else ''
                        )
        self.radius = self.commands['DIAM', 0].numbers[0] * self.unit / 2.0 if 'DIAM' in self.commands else np.inf
        self.stop = 'STOP' in self.commands
        self.distance = self.commands['DISZ', 0].numbers[0] * self.unit if 'DISZ' in self.commands else np.inf
        self.comment = self.commands['COMM', 0].argument if 'COMM' in self.commands else ''
        glass_name = (self.commands['GLAS', 0].words[0]
                      if 'GLAS' in self.commands and len(self.commands['GLAS', 0].words) > 0 else ''
                      )
        self.reflect = glass_name == 'MIRROR'  # Not 'MIRR' command for some reason
        self.clear_aperture_radius = (self.commands['CLAP', 0].numbers[1] * self.unit / 2.0
                                      if 'CLAP' in self.commands and len(self.commands['CLAP', 0].numbers) > 1
                                      else np.inf
                                      )
        if glass_name in {'', 'MIRROR'}:
            material = background_material
        else:
            material = None
            for material_library in material_libraries:
                if glass_name in material_library:
                    material = material_library.find_all(glass_name)[0]
                    break
            if material is None:
                log.error(f'Glass {glass_name} not found in {material_libraries}.')
                material = Material(name=glass_name)  # Dummy material
        self.material: Material = material
        self.floating_aperture = self.commands['FLAP', 0].numbers if 'FLAP' in self.commands else 0
        self.conic_constant = self.commands['CONI', 0].numbers if 'CONI' in self.commands else 0
        self.parameters = [_.numbers[0] for _ in self.commands.sort_and_merge('PARM')]
        self.extra_data = self.commands['XDAT', 0].numbers if 'XDAT' in self.commands else []
        self.aperture_offsets = self.commands['OBDC', 0].numbers[:2] if 'OBDC' in self.commands else []
        pickup_parameter_commands = self.commands['PPAR']  # of the form PPAR parameter from_surface factor offset 0
        for pickup_parameter_command in pickup_parameter_commands:
            parameter_index, from_surface, factor, offset = pickup_parameter_command.numbers[:4]
            parameter_index -= 1
            # self.parameter[parameter_index] = surface[from_surface].parameters[parameter_index] * factor + offset
            # The number on file seem to be computed already.

        def standard_sag(r2: array_like) -> array_type:
            return self.curvature * r2 / (1 + (1 - (1 + self.conic_constant) * self.curvature ** 2 * r2)**0.5)

        def odd_asphere_sag(r2: array_like, coefficients: array_like) -> array_type:
            sag = 0
            r = r2 ** 0.5
            for _, c in enumerate(coefficients):
                sag = sag + c * (r ** (_ + 1))
            return standard_sag(r2) * sag

        def even_asphere_sag(r2: array_like, coefficients: array_like) -> array_type:
            sag = 0
            for _, c in enumerate(coefficients):
                sag = sag + c * (r2 ** (_ + 1))
            return standard_sag(r2) * sag

        def zernike_sag(position: array_like, coefficients: array_like, indices: array_like = tuple(),
                        radius: array_type = 1.0,
                        ) -> array_type:
            position = asarray(position)
            rho = norm(position[..., :2]) / radius
            phi = np.arctan2(position[..., 1], position[..., 0])
            z = zernike.Polynomial(coefficients=coefficients, indices=indices)
            return z(rho, phi)

        def poly_sag(position: array_type, coefficients: array_like, radius: array_type = 1.0) -> array_type:
            p = asarray(position) / radius
            sag = 0
            for _, c in enumerate(coefficients):
                if c != 0:
                    # 0: x^1 y^0, 1: x^0 y^1,    2: x^2 y^0, 3: x^1 y^1, 4: x^0 y^2,  5: x^3 y^0, ...
                    # j = n * (n+1) / 2 - 1 -> n * (n+1) / 2 - 1 + n
                    nb_factors = np.floor((2 * (_ + 1)) ** 0.5).astype(int)
                    exponent_y = _ - nb_factors * (nb_factors + 1) // 2 + 1
                    exponents = [nb_factors - exponent_y, exponent_y]
                    sag = sag + c * (p[..., 0] ** exponents[0]) * (p[..., 1] ** exponents[1])
            return sag

        def radial_squared(_: array_like) -> array_type:
            return np.sum(asarray(_)[..., :2] ** 2)

        match self.type:
            case 'BICONICX':
                log.warning(
                    f'Surface {self.type}({self.parameters}, {self.extra_data}) not implemented:\n{self.commands}',
                )
            case 'BLACKBOX':  # A blackbox element, described by a binary file, likely with extension .zbb
                self.blackbox_filename = self.comment
            case 'COORDBRK':  # A Transform, not a Surface
                self.decenter_xy = asarray(self.parameters[:2]) * self.unit

                # Rotation of the coordinate system of the optics. Euler angles are applied in the order x, y, z.
                self.euler_angles = [_ * np.pi / 180.0 for _ in self.parameters[2:5]]

                # If False, decenter, then apply Euler angles; and then translate thickness.
                # If True, apply Euler angles; then decenter and thickness.
                self.rotate_before_decenter = self.parameters[5] != 0
            case 'EVENASPH':  # Even Asphere Surface, used as the basis for many other surfaces
                self.sag = lambda pos: even_asphere_sag(radial_squared(pos), asarray(self.parameters) * self.unit)
                # In lens units, the extended version uses normalized radii, rho
            case 'PARAXIAL':
                self.focal_length = self.parameters[0] * self.unit
                opd_calc_mode = self.parameters[1]
            case 'STANDARD':
                # STANDARD surface: z**2 == 2 * r * self.curvature - (1 + self.conic_constant) * r**2,
                # used as the basis for many other surfaces
                self.sag = lambda position: standard_sag(radial_squared(position))
            case 'SZERNSAG':  # Zernike Standard Sag Surface is derived from the Even aspheric surface
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * self.unit
                self.sag = lambda position: (
                    even_asphere_sag(radial_squared(position), asarray(self.parameters) * self.unit) +
                    zernike_sag(position, self.zernike_coefficients, radius=self.zernike_radius),
                )
            case 'SZERNPHA':  # STANDARD sag, but opd changed by Zernikes
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * 2 * np.pi
                self.sag = lambda pos: even_asphere_sag(radial_squared(pos), asarray(self.parameters) * self.unit)
                self.phase = lambda pos: zernike_sag(pos, self.zernike_coefficients, radius=self.zernike_radius)
            case 'FZERNSAG':  # Zernike Fringe Sag Surface is derived from the Even aspheric surface
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * self.unit
                self.sag = lambda position: (
                    even_asphere_sag(radial_squared(position), asarray(self.parameters) * self.unit) +
                    zernike_sag(
                        position, self.zernike_coefficients,
                        indices=zernike.fringe2index(range(1, 1 + len(self.zernike_coefficients))),
                        radius=self.zernike_radius,
                    )
                )
            case 'FZERNPHA':  # STANDARD sag, but opd changed by Zernikes
                self.zernike_radius = self.extra_data[1] * self.unit
                self.zernike_coefficients = asarray(self.extra_data[2:]) * 2 * np.pi
                self.sag = lambda pos: even_asphere_sag(radial_squared(pos), asarray(self.parameters) * self.unit)
                self.phase = lambda position: zernike_sag(
                    position, self.zernike_coefficients,
                    indices=zernike.fringe2index(range(1, 1 + len(self.zernike_coefficients))),
                    radius=self.zernike_radius,
                )
            case 'GRID_SAG':
                log.warning(
                    f'Surface {self.type}({self.parameters}, {self.extra_data}) not implemented:\n{self.commands}',
                )
            case 'IRREGULA':
                log.warning(
                    f'Surface {self.type}({self.parameters}, {self.extra_data}) not implemented:\n{self.commands}',
                )
            case 'TILTSURF':
                # A planar surface with a rotation before and its reverse after.
                self.euler_angles = np.arctan(self.parameters[:2])  # TODO wrap in Transforms
                self.sag = lambda position: np.zeros_like(position[..., 0])
            case 'TOROIDAL':
                # self.parameters has [extrapolate_zernike, radius_of_rotation, coefficients]
                # data_x [nb_zernikes, norm_radius, *zernike_terms], and VPAR also seems to contain some info?
                def toroidal_sag(p: array_type) -> array_like:
                    p = asarray(p)
                    z_in_plane = even_asphere_sag(p[..., 1] ** 2, self.parameters[2:])
                    radius_of_rotation = self.parameters[1]  # if self.parameters[1] != 0 else np.inf
                    curved_in_x = radius_of_rotation != 0
                    z_to_origin_sqd = (z_in_plane - radius_of_rotation) ** 2 - p[..., 1] ** 2
                    toroid = (z_to_origin_sqd ** 0.5 + radius_of_rotation
                              ) * curved_in_x + (1 - curved_in_x) * z_in_plane
                    return toroid + zernike_sag(p,
                                                coefficients=asarray(self.extra_data[2:]) * self.unit,
                                                radius=self.extra_data[1] * self.unit,
                                                )
                self.sag = toroidal_sag
            case 'XPOLYNOM':
                self.sag = lambda position: (
                    standard_sag(radial_squared(position)) +
                    poly_sag(position, self.extra_data[2:], radius=self.extra_data[1] * self.unit),
                )
            case _:
                log.warning(f"Unknown surface type: '{self.type}'!")

        super().__init__(
            interface=SnellInterface(transform=SphericalTransform(curvature=self.curvature)),
            aperture=DiskAperture(outer_radius=self.radius),
            transform=Translation([0, 0, self.distance]),
        )

    @property
    def eccentricity(self):
        """
        The eccentricity, i.e. the deviation from spherical, of this surface.

        It is the square root of the negated conic constant.
        """
        return (-self.conic_constant)**0.5

    def __str__(self) -> str:
        """
        The text string from which this object is parsed.

        Aside from the line-break character choice, this should correspond to the input at creation using from_str().
        """
        return str(self.commands)

    def __repr__(self) -> str:
        """The representation of this surface is its constructor, including the commands it is based on."""
        return f'{self.__class__.__name__}(commands={repr(self.commands)})'
