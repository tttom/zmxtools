import logging
import numpy as np
import numpy.testing as npt
from collections import defaultdict
from typing import Type, List
import itertools
import pathlib

from tests.agf import test_directory, test_files
from zmxtools.definitions import const_c
from zmxtools import agf

from tests.zmx import log
log = log.getChild(__name__)

log.level = logging.DEBUG
agf.log.level = logging.DEBUG


def test_from_file():
    """Tests agf.AgfMaterialLibrary.from_file function."""

    assert len(test_files) > 1, (
        f'No agf files found in {test_directory}!'
    )

    material_type_dict = defaultdict[Type, List[pathlib.Path]](list)
    tested_material_type_dict = defaultdict[Type, List[pathlib.Path]](list)
    for agf_file_path in test_files:
        log.info(f"Testing {agf_file_path}...")
        material_library = agf.AgfMaterialLibrary.from_file(agf_file_path)
        log.info(f"Read {agf_file_path}.")

        assert len(material_library.name) > 0, f"No name detected for the material library in {agf_file_path}!"
        assert len(material_library) >= 1, f"Too few materials ({len(material_library)}) found in library {material_library.name} at {agf_file_path}!"

        for material in material_library:
            material_type_dict[material.__class__].append(agf_file_path)

            if material.name == "N-BK7":  # 2 in SCHOTT
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=656.281e-9), 1.5143, decimal=4,
                                        err_msg=f"{material.name} n at 656.281nm is returned incorrectly.")
                npt.assert_almost_equal(material.refractive_index_C, 1.5143, decimal=4,
                                        err_msg=f"{material.name} n_C is returned incorrectly as {material.refractive_index_C}.")
                npt.assert_almost_equal(material.refractive_index_d, 1.5168, decimal=4,
                                        err_msg=f"{material.name} n_d is returned incorrectly as {material.refractive_index_d}.")
                npt.assert_almost_equal(material.refractive_index_F, 1.5224, decimal=4,
                                        err_msg=f"{material.name} n_F is returned incorrectly as {material.refractive_index_F}.")
                npt.assert_almost_equal(material.constringence, 64.17, decimal=2,
                                        err_msg=f"{material.name} Abbe number / constringence is returned incorrectly as {material.constringence}.")
                npt.assert_almost_equal(material.refractive_index(wavelength=500e-9), 1.5214, decimal=4,
                                        err_msg=f"Refractive index is incorrect.")
                npt.assert_almost_equal(material.refractive_index(wavenumber=2 * np.pi / 500e-9), 1.5214, decimal=4,
                                        err_msg=f"Refractive index is incorrect when using wavenumber.")
                npt.assert_almost_equal(material.refractive_index(angular_frequency=2 * np.pi / 500e-9 * const_c), 1.5214, decimal=4,
                                        err_msg=f"Refractive index is incorrect when using angular frequency.")
                npt.assert_almost_equal(material.transmittance(0.025, wavelength=500e-9), 0.9940, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 25mm")
                npt.assert_almost_equal(material.extinction_coefficient(wavelength=500e-9), 9.5781e-9, decimal=13,
                                        err_msg=f"extinction coefficient κ is incorrect.")
                npt.assert_almost_equal(material.extinction_coefficient(wavenumber=2 * np.pi / 500e-9), 9.5781e-9, decimal=13,
                                        err_msg=f"extinction coefficient κ is incorrect when using wavenumber.")
                npt.assert_almost_equal(material.extinction_coefficient(angular_frequency=2 * np.pi / 500e-9 * const_c), 9.5781e-9, decimal=13,
                                        err_msg=f"extinction coefficient κ is incorrect when using angular frequency.")
                npt.assert_almost_equal(material.transmittance(1.0, wavelength=500e-9), 0.78606, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 1m")
                npt.assert_almost_equal(material.transmittance(10e-3, wavelength=500e-9), 0.99760, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 1cm")
                npt.assert_almost_equal(material.transmittance(1e-3, wavelength=500e-9), 0.99976, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 1mm")
                npt.assert_almost_equal(material.absorption_coefficient(wavelength=500e-9), 0.78606, decimal=5,
                                        err_msg=f"{material.name} absorption coefficient incorrect")
                if isinstance(material, agf.AgfMaterial):
                    npt.assert_almost_equal(material.density, 2.51e3, decimal=5,
                                            err_msg=f"{material.name} density is returned incorrectly as {material.density}.")
                else:
                    raise AssertionError(f"{material} should be an subclass of agf.AgfMaterial.")

            if material.name == "S-BAH54":  # 1 in OHARA
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=480e-9), 1.7076, decimal=4,
                                        err_msg=f"{material.name} n at 480nm is returned incorrectly.")
                npt.assert_almost_equal(material.constringence, 42.17, decimal=2,
                                        err_msg=f"{material.name} Abbe number / constringence is returned incorrectly as {material.constringence}.")
                npt.assert_almost_equal(material.transmittance(10e-3, wavelength=480e-9), 0.99, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 10mm")

            if material.name == "S-BAH11":  # 2 in OHARA
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=488e-9), 1.6761, decimal=4,
                                        err_msg=f"{material.name} n at 488nm is returned incorrectly.")
                npt.assert_almost_equal(material.constringence, 48.32, decimal=2,
                                        err_msg=f"{material.name} Abbe number / constringence is returned incorrectly as {material.constringence}.")
                npt.assert_almost_equal(material.transmittance(10e-3, wavelength=488e-9), 0.99223, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 10mm")

            if material.name == "H-FK61" and material_library.name == "cdgm.agf":  # 2
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=633e-9), 1.4957, decimal=4,
                                        err_msg=f"{material.name} n at 633nm is returned incorrectly.")
                npt.assert_almost_equal(material.refractive_index(wavelength=350e-9), 1.5139, decimal=4,
                                        err_msg=f"{material.name} n at 350nm is returned incorrectly.")
                npt.assert_almost_equal(material.extinction_coefficient(wavelength=633e-9), 1.0084e-8, decimal=12,
                                        err_msg=f"extinction coefficient κ is incorrect.")
                npt.assert_almost_equal(material.transmittance(10e-3, wavelength=633e-9), 0.99800, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 1cm")

            if material.name == "NICHIA_MELT1":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index_d, 1.493724, decimal=6,
                                        err_msg=f"{material.name} n at 500nm is returned incorrectly.")
                npt.assert_almost_equal(material.transmittance(25e-3, wavelength=500e-9), 1.0, decimal=5,
                                        err_msg=f"{material.name} transmittance incorrect for 25mm")

            if material.name == "BD2":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=11e-6), 2.5983, decimal=4,
                                        err_msg=f"{material.name} n is returned incorrectly.")
                # npt.assert_almost_equal(material.absorption_coefficient(wavelength=11e-6), 0.03, decimal=3,
                #                         err_msg=f"{material.name} absorption coefficient incorrect")

            if material.name == "KRS5":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=1e-6), 2.4462, decimal=4,
                                        err_msg=f"{material.name} n is returned incorrectly.")

            if material.name == "CALCITE":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=0.5e-6), 1.6660, decimal=3,  # TODO: Can this be more accurate?
                                        err_msg=f"{material.name} n is returned incorrectly.")

            if material.name == "YVO4":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=1e-6), 1.9603, decimal=3,  # TODO: Can this be more accurate?
                                        err_msg=f"{material.name} n is returned incorrectly.")

            if material.name == "ZNO":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=1e-6), 1.9433, decimal=4,
                                        err_msg=f"{material.name} n is returned incorrectly.")

            if material.name == "AGGAS2":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=1e-6), 2.4583, decimal=2,  # TODO: Can this be more accurate?
                                        err_msg=f"{material.name} n is returned incorrectly.")

            if material.name == "J-FK5":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index_d, 1.487490, decimal=6,
                                        err_msg=f"{material.name} n is returned incorrectly.")
                npt.assert_almost_equal(material.transmittance(10e-3, wavelength=500e-9), 0.997, decimal=4,
                                        err_msg=f"{material.name} absorption coefficient incorrect")

            if material.name == "E-LASFH6":
                tested_material_type_dict[material.__class__].append(agf_file_path)
                log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
                npt.assert_almost_equal(material.refractive_index(wavelength=500e-9), 1.8202, decimal=4,
                                        err_msg=f"{material.name} n is returned incorrectly.")
                npt.assert_almost_equal(material.extinction_coefficient(wavelength=500e-9), 6.8222e-8, decimal=4,
                                        err_msg=f"{material.name} extinction coefficient incorrect")

            # if material.name == "CDASS":
            #     tested_material_type_dict[material.__class__].append(agf_file_path)
            #     conrady_files.append(agf_file_path.name)
            #     log.info(f"Analysing {material.name} of type {material.__class__.__name__} from {material_library.name}...")
            #     npt.assert_almost_equal(material.refractive_index_d, 3.5963, decimal=6,
            #                             err_msg=f"{material.name} n at 500nm is returned incorrectly.")
            #     npt.assert_almost_equal(material.transmittance(25e-3, wavelength=500e-9), 1.0, decimal=5,
            #                             err_msg=f"{material.name} transmittance incorrect for 25mm")

    all_classes = set(material_type_dict.keys())
    log.info(f"All classes: {', '.join(str(_.__name__) for _ in all_classes)}")

    tested_classes = set(tested_material_type_dict.keys())
    log.info(f"Tested classes: {', '.join(str(_.__name__) for _ in tested_classes)}")
    untested_classes = all_classes.difference(tested_classes)
    log.info(f"Untested classes: {', '.join(str(_.__name__) for _ in untested_classes)}")
    used_catalogs = set(itertools.chain.from_iterable(tested_material_type_dict[_] for _ in tested_classes))
    log.info(f"Used catalogs: {', '.join(_.name for _ in used_catalogs)}")

    for t, p in material_type_dict.items():
        if t not in tested_classes:
            log.info(f"Did not test {t.__name__}: {[_.name for _ in set(p)]}")

    for _ in (agf.SchottAgfMaterial, agf.Sellmeier1AgfMaterial, agf.Sellmeier3AgfMaterial,
              agf.Sellmeier4AgfMaterial, agf.Sellmeier5AgfMaterial,
              agf.HandbookOfOptics1AgfMaterial, agf.HandbookOfOptics2AgfMaterial,
              agf.ConradyAgfMaterial, agf.HerzbergerAgfMaterial,
              agf.Extended2AgfMaterial,
              agf.Formula13AgfMaterial):
        assert _ in tested_classes, f"Did not test material of type {_.__class__}"
        # TODO: Test agf.Sellmeier2AgfMaterial, agf.Extended1AgfMaterial

