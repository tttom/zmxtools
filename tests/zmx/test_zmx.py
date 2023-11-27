from typing import List
import math
from tests.zmx import test_directory, test_files
from zmxtools import zmx
from zmxtools.definitions import vacuum

from tests.zmx import log
log = log.getChild(__name__)


def test_read():
    """Tests the zmxtools.zmx.read function."""

    assert len(test_files) > 1, (
        f'No zmx files found in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zmx_file_path in test_files:
        log.info(f"Testing {zmx_file_path}...")
        optical_system = zmx.read(zmx_file_path)
        print(optical_system)
        assert len(optical_system.name) > 0, f"No name detected for the optical system in {zmx_file_path}!"
        assert len(optical_system.description) > 0, f"No description detected for the optical system in {zmx_file_path}!"
        print(optical_system.surfaces)
        assert len(optical_system.surfaces) > 0, f"No surfaces detected for the optical system in {zmx_file_path}!"
        assert len(optical_system.surfaces) >= 4, f"Only {len(optical_system.surfaces)} surfaces detected for the optical system in {zmx_file_path}. At least 4 expected."
        assert optical_system.unit == 1e-3, f"Unit is {optical_system.unit}, expected 1e-3 for millimeters."
        nb_stops = sum(_.stop for _ in optical_system.surfaces)
        assert nb_stops <= 1, f"Multiple stop surfaces set! At most one of the {optical_system.surfaces} surfaces should have a stop."
        assert nb_stops >= 1, f"No stop surface set! At least one of the {optical_system.surfaces} surfaces is expected for the test lens files."
        assert not optical_system.surfaces[0].stop, "The stop should not be set at the object surface."
        assert not optical_system.surfaces[-1].stop, "The stop should not be set at the image surface."
        distances = [_.distance for _ in optical_system.surfaces]
        assert all(abs(_) != math.inf  for _ in distances[1:]), \
            f"With the exception of that of the first surface, lens elements should not be infinitely thick, only the object surface is expected to have infinite thickness, not {distances}."
        curvatures = [_.curvature for _ in optical_system.surfaces]
        assert all(abs(_) != math.inf for _ in curvatures), f"The surface curvatures should all be finite, not {curvatures}."
        radii = [_.radius for _ in optical_system.surfaces]
        assert all(0.0 < _ for _ in radii[1:]), f"With the exception of that of the first surface, all radii should be strictly positive, not {radii}."
        materials = [_.material for _ in optical_system.surfaces]
        assert max(len(_.name) for _ in materials if _ != vacuum) > 2, "Glass names are too short."

