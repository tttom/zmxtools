import logging
import math

from numpy import testing as npt

from tests.zmx import log, test_agf_files, test_directory, test_zmx_files
from zmxtools import zmx
from zmxtools.optical_design.material import Vacuum

log = log.getChild(__name__)
log.level = logging.DEBUG
zmx.log.level = logging.WARNING

vacuum = Vacuum()


def check_optical_design(optical_design, file_path):
    """Checks whether an optical design has plausible properties."""
    npt.assert_equal(len(optical_design.name) > 0, desired=True,
                     err_msg=f'No name detected for the optical system in {file_path}!',
                     )
    zmx_surfaces = optical_design.surfaces
    has_blackbox = any(_.type == 'BLACKBOX' for _ in zmx_surfaces)
    if not has_blackbox:
        assert len(optical_design.description) > 0, f'No NOTE detected for the optical system in {file_path}!'

    npt.assert_equal(isinstance(optical_design, zmx.ZmxOpticalDesign), desired=True,
                     err_msg=f'The optical design {repr(optical_design)} is not a ZmxOpticalDesign.',
                     )
    npt.assert_equal(len(zmx_surfaces) > 0, desired=True,
                     err_msg=f'No surfaces detected for the optical system in {file_path}!',
                     )
    npt.assert_equal(len(zmx_surfaces) >= 3, desired=True,
                     err_msg=f'Only {len(zmx_surfaces)} surfaces detected for the optical system in {file_path}.' +
                             f'At least 3 expected in {file_path}.',
                     )
    npt.assert_equal(optical_design.unit in {1e-3, 10e-3, 0.0254}, desired=True,
                     err_msg=f'Unit is {optical_design.unit}, expected 1e-3 for millimeters in {file_path}.',
                     )
    nb_stops = sum(_.stop for _ in zmx_surfaces)
    npt.assert_equal(nb_stops <= 1, desired=True,
                     err_msg='Multiple stop surfaces set! At most one of the ' +
                             f'{zmx_surfaces} surfaces should have a stop in {file_path}.',
                     )
    npt.assert_equal(nb_stops >= 1, desired=True,
                     err_msg='No stop surface set! At least one of the ' +
                             f'{zmx_surfaces} surfaces is expected for the test lens files in {file_path}.',
                     )
    npt.assert_equal(optical_design.source.surface.stop, desired=False,
                     err_msg='The stop should not be at the object surface in {zmx_file_path}.',
                     )
    npt.assert_equal(zmx_surfaces[-1].stop, desired=False,
                     err_msg='The stop should not be set at the image surface in {zmx_file_path}.',
                     )
    distances = [_.distance for _ in zmx_surfaces]
    assert all(abs(_) != math.inf for _ in distances[1:-2]), (
        'With the exception of that of the first surface, lens elements should not be infinitely thick, only the ' +
        'object and image surfaces may have infinite thickness, not ' +
        f"{[f'{_.distance} {_.type}' for _ in zmx_surfaces[1:-2]]} in {file_path}."
    )
    total_track = sum(distances[1:-1])
    npt.assert_equal(1e-3 <= total_track <= 1.0, desired=True,
                     err_msg=f'Total track {total_track} too extreme in {file_path}.',
                     )
    curvatures = [_.curvature for _ in zmx_surfaces]
    npt.assert_equal(all(abs(_) != math.inf for _ in curvatures), desired=True,
                     err_msg=f'The surface curvatures should all be finite, not {curvatures} in {file_path}.',
                     )
    npt.assert_equal(has_blackbox or any(_ != 0 for _ in curvatures), desired=True,
                     err_msg=f'No curved surfaces found in {file_path}.',
                     )
    npt.assert_equal(any(0 < _.radius for _ in zmx_surfaces), desired=True,
                     err_msg='At least some radii are expected to be strictly positive, not' +
                             f" {[f'{_.radius} {_.type}' for _ in zmx_surfaces]} in {file_path}.",
                     )
    npt.assert_equal(len(optical_design.material_libraries) > 0, desired=True,
                     err_msg=f'No material libraries specified in {file_path}.',
                     )
    materials = [_.material for _ in zmx_surfaces]
    if len([_.type == 'PARAXIAL' for _ in zmx_surfaces]) == 0:
        npt.assert_equal(len([_ for _ in materials if _ != vacuum]) > 0, desired=True,
                         err_msg=f'No non-vacuum materials found: {materials} in {file_path}',
                         )
        npt.assert_equal(
            max(len(_.name) for _ in materials if _ != vacuum) >= 2, desired=True,
            err_msg=f'One of the glass names is too short: {[_.name for _ in materials if _ != vacuum]} in {file_path}',
        )

    if file_path.parent.name == 'long_wavelength':
        npt.assert_equal(all(100e-6 <= _ <= 10e-3 for _ in optical_design.source.wavelengths), desired=True,
                         err_msg=f'Unusual long wavelengths found {optical_design.source.wavelengths} in {file_path}!',
                         )
    else:
        npt.assert_equal(all(10e-9 <= _ <= 100e-6 for _ in optical_design.source.wavelengths), desired=True,
                         err_msg=f'Unusual wavelengths found {optical_design.source.wavelengths} in {file_path}!',
                         )


def test_from_file():
    """Tests the zmx.ZmxOpticalDesign.from_file function."""
    assert len(test_zmx_files) > 1, (
        f'No zmx files found in {test_directory}!'
    )

    for zmx_file_path in test_zmx_files:
        log.info(f'Testing {zmx_file_path}...')
        optical_design = zmx.ZmxOpticalDesign.from_file(zmx_file_path, test_agf_files)
        log.info(f'Read {zmx_file_path}.')

        if not optical_design.sequential or zmx_file_path.parent.name in {'no_name', 'no_note'}:
            check_optical_design(optical_design, zmx_file_path)


if __name__ == '__main__':
    optical_design = zmx.ZmxOpticalDesign.from_file(test_zmx_files[0], test_agf_files)
    wavelength = optical_design.source.wavelengths.ravel()[0]
    for surface in optical_design.surfaces:
        log.info(surface.type + (' STOP' if surface.stop else '') + ' surface with ' +
                 ('∞' if surface.curvature == 0 else f'{1 / surface.curvature / 1e-3:0.3f} mm') +
                 f' radius of curvature and {surface.radius * 2 / 1e-3:0.3f} mm aperture.',
                 )
        if surface.distance != 0:
            log.info(f'  {surface.distance / 1e-3:0.3f} mm spacing with {surface.material.name}' +
                     f' n={surface.material.complex_refractive_index(wavelength=wavelength)}' +
                     f' @ {wavelength / 1e-9:0.1f}nm.',
                     )
