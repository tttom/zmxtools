import pathlib

import numpy as np
from numpy import testing as npt

from tests.agf import test_directory
from zmxtools import agf

glass_directory = test_directory / "glass"

catalog_files: list[pathlib.Path] = sorted(glass_directory.glob("*.agf"))

# Infrared crystals, birefringent materials, and some model glasses legitimately
# fall outside the visible-glass index range, or extrapolate wildly at the d-line.
SPECIAL_CATALOGS = {"birefringent", "cdgm", "infrared", "lightpath", "misc"}


def test_catalogs_present():
    assert len(catalog_files) >= 15, f"Too few glass catalogs in {glass_directory}: {catalog_files}."


def test_load_catalogs():
    """Every glass catalog loads and yields materials with plausible d-line indices."""
    for catalog_file in catalog_files:
        material_library = agf.AgfMaterialLibrary.from_file(catalog_file)
        assert len(material_library) >= 1, f"No materials found in {catalog_file}."
        assert len(material_library.name) > 0, f"No name found for the library in {catalog_file}."
        indices = [_.refractive_index_d for _ in material_library]
        finite = [float(_) for _ in indices if np.isfinite(_)]
        assert len(finite) >= 1, f"No material with a finite d-line index in {catalog_file}."
        if catalog_file.stem not in SPECIAL_CATALOGS:
            npt.assert_array_less(1.2, np.array(finite), err_msg=f"Very small refractive index in {catalog_file}.")
            npt.assert_array_less(np.array(finite), 2.3, err_msg=f"Very large refractive index in {catalog_file}.")


def test_catalog_materials_resolve():
    """Spot-check that glasses can be found by name in the new catalogs."""
    material_library = agf.AgfMaterialLibrary.from_file(glass_directory / "hoya.agf")
    assert len(material_library.find_all("FD")) > 0, "No FD-series flint glasses found in hoya.agf."
