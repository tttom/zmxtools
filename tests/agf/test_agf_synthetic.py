from pathlib import Path

import numpy as np
import pytest
from numpy import testing as npt

from zmxtools import agf

SELLMEIER1 = """NM N-BK7 2 1.5168 64.17 0 1 1 1
  LD 0.3 2.5
  CD 2.271181 0.01 1.070809 0.02 0.01104338 0.03
  ED 0 0 2.51 0 0
  IT 0.38 1 0.005
  IT 2.5 1 0.01
  GC 1 1 BAK1
  OD 1 1 1 1 1 1
"""

EXTENDED = """NM E-GLASS 10 1.688 53.27 0 1 1 1
  LD 0.3 2.5
  CD 2.3047 0.01 1.0 0.02 1.0e-2 0.03
  ED 0 0 3.0 0 0
"""

SELLMEIER3 = """NM S3 6 1.5 60.0 0 1 1 1
  LD 0.3 2.5
  CD 1.0 0.01 1.0 0.01 1.0 0.01
  ED 0 0 2.5 0 0
"""

SELLMEIER4 = """NM S4 9 1.5 60.0 0 1 1 1
  LD 0.3 2.5
  CD 1.0 0.01 1.0 0.01 1.0 0.01 1.0 0.01
  ED 0 0 2.5 0 0
"""

SELLMEIER5 = """NM S5 11 1.5 60.0 0 1 1 1
  LD 0.3 2.5
  CD 1.0 0.01 1.0 0.01 1.0 0.01 1.0 0.01 1.0 0.01
  ED 0 0 2.5 0 0
"""

NO_MATERIALS = """TITLE empty library
"""


class TestAgfSynthetic:
    def test_parse_sellmeier1(self):
        library = agf.AgfMaterialLibrary.from_str(SELLMEIER1)
        npt.assert_equal(len(library.materials), 1)
        glass = library.materials[0]
        npt.assert_equal(glass.name, "N-BK7")
        n = glass.refractive_index(wavelength=587.5618e-9)
        npt.assert_equal(1.0 < n < 3.0, True)
        npt.assert_equal(str(library).startswith("NM"), True)
        npt.assert_equal(repr(library).startswith("AgfMaterialLibrary("), True)
        npt.assert_equal(str(glass).startswith("NM"), True)
        npt.assert_equal(repr(glass).startswith("Sellmeier1AgfMaterial("), True)

    def test_extends(self, subtests):
        for text, cls in (
            (EXTENDED, "Extended1AgfMaterial"),
            (SELLMEIER3, "Sellmeier3AgfMaterial"),
            (SELLMEIER4, "Sellmeier4AgfMaterial"),
            (SELLMEIER5, "Sellmeier5AgfMaterial"),
        ):
            with subtests.test(cls=cls):
                library = agf.AgfMaterialLibrary.from_str(text)
                npt.assert_equal(type(library.materials[0]).__name__, cls)
                n = library.materials[0].refractive_index(wavelength=587.5618e-9)
                npt.assert_equal(np.isfinite(n), True)

    def test_empty_library(self):
        library = agf.AgfMaterialLibrary.from_str(NO_MATERIALS)
        npt.assert_equal(len(library.materials), 0)

    def test_unknown_formula(self):
        with pytest.raises(ValueError):
            agf.AgfMaterialLibrary.from_str("NM BAD 99 1.5 60 0 1 1 1\n  LD 0.3 2.5\n  CD 1 1 1 0 0 0\n")

    def test_blank_lines(self):
        library = agf.AgfMaterialLibrary.from_str("\n" + SELLMEIER1 + "\n\n")
        npt.assert_equal(len(library.materials), 1)

    def test_name_from_file(
        self,
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/MYGLASS.agf"
            with open(path, "w", encoding="utf-8") as f:
                f.write(SELLMEIER1)
            library = agf.AgfMaterialLibrary.from_file(Path(path))
            npt.assert_equal(library.name, "MYGLASS.agf")

    def test_transmission(self):
        library = agf.AgfMaterialLibrary.from_str(SELLMEIER1)
        glass = library.materials[0]
        t = glass.transmittance(thickness=0.005, wavelength=587.5618e-9)
        npt.assert_equal(0 < t <= 1, True)

    def test_as_float(self):
        npt.assert_equal(agf.as_float("1.5e-3"), 1.5e-3)
