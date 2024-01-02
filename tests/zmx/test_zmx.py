from collections import defaultdict
import logging
import math
from tests.zmx import test_directory, test_zmx_files, test_agf_files
from zmxtools import zmx
from zmxtools.optical_design.material import Vacuum

from tests.zmx import log
log = log.getChild(__name__)

log.level = logging.DEBUG
zmx.log.level = logging.WARNING

vacuum = Vacuum()


def assert_optical_design(optical_design, file_path):
    assert len(optical_design.name) > 0, f"No name detected for the optical system in {file_path}!"
    assert len(optical_design.description) > 0, f"No note detected for the optical system in {file_path}!"

    assert len(optical_design.surfaces) > 0, f"No surfaces detected for the optical system in {file_path}!"
    assert len(optical_design.surfaces) >= 3, f"Only {len(optical_design.surfaces)} surfaces detected for the optical system in {file_path}. At least 3 expected in {file_path}."
    assert optical_design.unit in (1e-3, 10e-3, 0.0254), f"Unit is {optical_design.unit}, expected 1e-3 for millimeters in {file_path}."
    nb_stops = sum(_.stop for _ in optical_design.surfaces)
    assert nb_stops <= 1, f"Multiple stop surfaces set! At most one of the {optical_design.surfaces} surfaces should have a stop in {file_path}."
    assert nb_stops >= 1, f"No stop surface set! At least one of the {optical_design.surfaces} surfaces is expected for the test lens files in {file_path}."
    assert not optical_design.surfaces[0].stop, "The stop should not be set at the object surface in {zmx_file_path}."
    assert not optical_design.surfaces[-1].stop, "The stop should not be set at the image surface in {zmx_file_path}."
    distances = [_.distance for _ in optical_design.surfaces]
    assert all(abs(_) != math.inf for _ in distances[1:-2]), \
        ("With the exception of that of the first surface, lens elements should not be infinitely thick, only the "
         + f"object and image surfaces may have infinite thickness, not {[f'{_.distance} {_.type}' for _ in optical_design.surfaces[1:-2]]} in {file_path}.")
    total_track = sum(_.distance for _ in optical_design.surfaces[1:-1])
    assert 1e-3 <= total_track <= 1.0, f"Total track {total_track} too extreme in {file_path}."
    curvatures = [_.curvature for _ in optical_design.surfaces]
    assert all(abs(_) != math.inf for _ in curvatures), f"The surface curvatures should all be finite, not {curvatures} in {file_path}."
    assert any(_ != 0.0 for _ in curvatures), f"No curved surfaces found in {file_path}."
    assert any(0.0 < _.radius for _ in optical_design.surfaces), \
        f"At least some radii are expected to be strictly positive, not {[f'{_.radius} {_.type}' for _ in optical_design.surfaces]} in {file_path}."
    assert len(optical_design.material_libraries), f"No material libraries specified in {file_path}."
    materials = [_.material for _ in optical_design.surfaces]
    if len([_.type == "PARAXIAL" for _ in optical_design.surfaces]) == 0:
        assert len([_ for _ in materials if _ != vacuum]) > 0, f"No non-vacuum materials found: {materials} in {file_path}"
        assert max(len(_.name) for _ in materials if _ != vacuum) >= 2, \
            f"One of the glass names is too short: {[_.name for _ in materials if _ != vacuum]} in {file_path}"

    if file_path.parent.name != "long_wavelength":
        assert all(10e-9 <= _ <= 100e-6 for _ in optical_design.wavelengths), f"Unusual wavelengths found {optical_design.wavelengths} in {file_path}!"
    else:
        assert all(100e-6 <= _ <= 10e-3 for _ in optical_design.wavelengths), f"Unusual long wavelengths found {optical_design.wavelengths} in {file_path}!"


def test_from_file():
    """Tests the zmx.ZmxOpticalDesign.from_file function."""

    assert len(test_zmx_files) > 1, (
        f'No zmx files found in {test_directory}!'
    )

    for zmx_file_path in test_zmx_files:
        log.info(f"Testing {zmx_file_path}...")
        optical_design = zmx.ZmxOpticalDesign.from_file(zmx_file_path, test_agf_files)
        log.info(f"Read {zmx_file_path}.")

        if not optical_design.sequential or zmx_file_path.parent.name in ["no_name", "no_note"]:
            assert_optical_design(optical_design, zmx_file_path)
