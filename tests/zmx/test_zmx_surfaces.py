from typing import cast

import numpy as np
from numpy import testing as npt

from zmxtools import zmx
from zmxtools.optical_design.material import VACUUM

HEADER = """VERS 24.1
NAME Test system
AUTH Tester
NOTE 0 A synthetic test system
MODE SEQ
UNIT MM
WAVL 0.5875618
WWGT 1.0
"""


def build(surfaces: str, extra: str = "") -> zmx.ZmxOpticalDesign:
    """Build a minimal optical design from the given SURF sections."""
    return zmx.ZmxOpticalDesign.from_str(HEADER + extra + surfaces)


def standard_design() -> zmx.ZmxOpticalDesign:
    return build(
        """SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  GLAS __BLANK
SURF 1
  TYPE STANDARD
  CURV 0.01
  DISZ 5.0
  GLAS N-BK7 5168 6417
  CONI -1.0
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
""",
    )


class TestZmxSurfaces:
    def test_standard_sag(self):
        design = standard_design()
        surface = design.surfaces[1]
        npt.assert_equal(surface.type, "STANDARD")
        npt.assert_allclose(surface.curvature, 0.01 / 1e-3)
        npt.assert_allclose(surface.spacing, 5.0 * 1e-3)
        npt.assert_allclose(surface.eccentricity, 1.0)
        # On-axis sag is zero
        npt.assert_allclose(np.asarray(surface.sag(np.array([0.0, 0.0, 0.0]))), 0.0, atol=1e-15)
        # Standard sag: z = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2))
        r2 = 1e-4
        c = 0.01 / 1e-3
        expected = c * r2 / (1 + (1 - (1 + -1.0) * c**2 * r2) ** 0.5)
        npt.assert_allclose(np.asarray(surface.sag(np.array([1e-2, 0.0, 0.0]))), expected, rtol=1e-12)

    def test_even_asphere(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE EVENASPH
  CURV 0.01
  PARM 0 0.0
  PARM 1 1e-6
  PARM 2 2e-10
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_equal(surface.type, "EVENASPH")
        pos = np.array([1e-3, 0.0, 0.0])
        sag = surface.sag(pos)
        npt.assert_equal(np.isfinite(np.asarray(sag)), True)

    def test_odd_asphere(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE ODDASPH
  CURV 0.01
  PARM 0 1e-6
  PARM 1 2e-10
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        sag = surface.sag(np.array([1e-3, 0.0, 0.0]))
        npt.assert_equal(np.isfinite(np.asarray(sag)), True)

    def test_paraxial(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE PARAXIAL
  PARM 0 100.0
  PARM 1 0
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_allclose(surface.focal_length, 100.0 * 1e-3)

    def test_zernike_standard_sag(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE SZERNSAG
  CURV 0.01
  PARM 0.0
  XDAT 0 10.0 1e-3 2e-3 3e-3
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_allclose(surface.zernike_radius, 10.0 * 1e-3)
        sag = surface.sag(np.array([1e-3, 0.5e-3, 0.0]))
        npt.assert_equal(np.isfinite(np.asarray(sag)), True)

    def test_zernike_phase(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE SZERNPHA
  CURV 0.01
  PARM 0.0
  XDAT 0 10.0 1e-3 2e-3
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        phase = surface.phase(np.array([1e-3, 0.5e-3, 0.0]))
        npt.assert_equal(np.isfinite(phase), True)

    def test_fringe_zernike_sag(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE FZERNSAG
  CURV 0.01
  PARM 0.0
  XDAT 0 10.0 1e-3 2e-3 3e-3
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        sag = surface.sag(np.array([1e-3, 0.5e-3, 0.0]))
        npt.assert_equal(np.isfinite(np.asarray(sag)), True)

    def test_fringe_zernike_phase(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE FZERNPHA
  CURV 0.01
  PARM 0.0
  XDAT 0 10.0 1e-3 2e-3
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        phase = surface.phase(np.array([1e-3, 0.5e-3, 0.0]))
        npt.assert_equal(np.isfinite(phase), True)

    def test_tilt_surface(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE TILTSURF
  PARM 0 0.1
  PARM 1 0.2
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_allclose(surface.euler_angles, np.arctan([0.1, 0.2]))
        sag = surface.sag(np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))
        npt.assert_allclose(np.asarray(sag), 0.0)

    def test_toroidal(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE TOROIDAL
  PARM 0 0
  PARM 1 100.0
  PARM 2 0.0
  PARM 3 1e-4
  XDAT 0 10.0 1e-3
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        sag = surface.sag(np.array([1e-3, 2e-3, 0.0]))
        npt.assert_equal(np.isfinite(np.asarray(sag)), True)

    def test_x_polynomial(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE XPOLYNOM
  CURV 0.01
  XDAT 0 10.0 1e-3 2e-3 3e-3
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        sag = surface.sag(np.array([1e-3, 2e-3, 0.0]))
        npt.assert_equal(np.isfinite(np.asarray(sag)), True)

    def test_coord_break(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE COORDBRK
  PARM 0 1.0
  PARM 1 2.0
  PARM 2 3.0
  PARM 3 4.0
  PARM 4 5.0
  PARM 5 0
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_allclose(surface.decenter_xy, [1.0 * 1e-3, 2.0 * 1e-3])
        npt.assert_allclose(surface.euler_angles, [3.0 * np.pi / 180, 4.0 * np.pi / 180, 5.0 * np.pi / 180])
        npt.assert_equal(surface.rotate_before_decenter, False)

    def test_blackbox(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE BLACKBOX
  COMM my_blackbox.zbb
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_equal(surface.blackbox_filename, "my_blackbox.zbb")

    def test_not_implemented_types(self, subtests):
        for surface_type in ("BICONICX", "GRID_SAG", "IRREGULA", "SOMETHINGNEW"):
            with subtests.test(surface_type=surface_type):
                design = build(
                    f"""SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE {surface_type}
  PARM 0 1.0
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
                )
                npt.assert_equal(design.surfaces[1].type, surface_type)

    def test_mirror(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE STANDARD
  CURV 0.01
  GLAS MIRROR
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_equal(surface.is_mirror, True)
        npt.assert_equal(surface.material, VACUUM)

    def test_pickup_parameter(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE STANDARD
  CURV 0.01
  PPAR 1 0 1.0 0.0 0
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        npt.assert_equal(len(design.surfaces), 3)

    def test_obdc_and_flap(self):
        design = build(
            """SURF 0
  TYPE STANDARD
  CURV 0.0
SURF 1
  TYPE STANDARD
  CURV 0.01
  OBDC 1.0 2.0 0
  FLAP 1
SURF 2
  TYPE STANDARD
  CURV 0.0
""",
        )
        surface = design.surfaces[1]
        npt.assert_allclose(surface.aperture_offsets, [1.0, 2.0])
        npt.assert_allclose(surface.floating_aperture, 1)


class TestZmxOpticalDesign:
    def test_surfaces_and_media(self):
        design = standard_design()
        npt.assert_equal(len(design.surfaces), 3)
        npt.assert_equal(design.sequential, True)
        npt.assert_equal(design.name, "Test system")
        npt.assert_equal(design.author, "Tester")
        npt.assert_allclose(design.unit, 1e-3)
        npt.assert_allclose(cast("zmx.ZmxSource", design.source).wavelengths, 0.5875618 * 1e-6)
        npt.assert_equal(str(design).startswith("VERS"), True)
        npt.assert_equal(repr(design).startswith("ZmxOpticalDesign("), True)

    def test_non_sequential(self):
        design = zmx.ZmxOpticalDesign.from_str(HEADER.replace("MODE SEQ", "MODE NSC") + "SURF 0\nSURF 1\n")
        npt.assert_equal(design.sequential, False)

    def test_unknown_mode(self):
        design = zmx.ZmxOpticalDesign.from_str(HEADER.replace("MODE SEQ", "MODE XYZ") + "SURF 0\nSURF 1\n")
        npt.assert_equal(design.sequential, False)

    def test_units(self, subtests):
        for code, expected in (("IN", 25.4e-3), ("FEET", 304.8e-3), ("METER", 1.0), ("KM", 1e3)):
            with subtests.test(code=code):
                design = zmx.ZmxOpticalDesign.from_str(HEADER.replace("UNIT MM", f"UNIT {code}") + "SURF 0\nSURF 1\n")
                npt.assert_allclose(design.unit, expected)

    def test_unknown_unit(self):
        design = zmx.ZmxOpticalDesign.from_str(HEADER.replace("UNIT MM", "UNIT PARSEC") + "SURF 0\nSURF 1\n")
        npt.assert_allclose(design.unit, 1.0)

    def test_wavm(self):
        design = zmx.ZmxOpticalDesign.from_str(
            HEADER.replace("WAVL 0.5875618\nWWGT 1.0\n", "")
            + """WAVM 0 0.486 0.1
WAVM 1 0.587 1.0
WAVM 2 0.656 0.1
WAVM 3 0.656 0.2
SURF 0
SURF 1
""",
        )
        npt.assert_allclose(sorted(cast("zmx.ZmxSource", design.source).wavelengths.ravel()), [0.486e-6, 0.587e-6])

    def test_field_configuration(self):
        design = zmx.ZmxOpticalDesign.from_str(
            HEADER
            + """FNUM 5.6
EFFL 100.0
ENPD 10.0
OBNA 0.1
FTYP 1
FCOM field comment
SURF 0
SURF 1
""",
        )
        npt.assert_allclose(design.numerical_aperture_image, 2.0 / 5.6)
        npt.assert_allclose(design.numerical_aperture_object, 0.1)
        npt.assert_equal(design.field_as_height, True)
        npt.assert_equal(design.field_at_image, False)
        npt.assert_equal(design.field_comment, "field comment")

    def test_coatings(self):
        design = zmx.ZmxOpticalDesign.from_str(
            HEADER
            + """COFN QF COATING1.DAT COATING2.DAT
SURF 0
SURF 1
""",
        )
        npt.assert_array_equal(design.coating_filenames, ["COATING1.DAT", "COATING2.DAT"])

    def test_missing_fields(self):
        design = zmx.ZmxOpticalDesign.from_str("SURF 0\nSURF 1\n")
        npt.assert_equal(design.version, "")
        npt.assert_equal(design.name, "")
        npt.assert_equal(design.author, "")
        npt.assert_equal(design.description, "")
        npt.assert_allclose(design.unit, 1.0)

    def test_str_roundtrip(self):
        design = standard_design()
        reparsed = zmx.ZmxOpticalDesign.from_str(str(design))
        npt.assert_equal(len(reparsed.surfaces), len(design.surfaces))
