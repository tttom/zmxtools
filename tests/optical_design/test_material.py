import unittest

import numpy as np
from numpy import testing as npt

from zmxtools.optical_design.material import (
    VACUUM,
    CiddorAir,
    FunctionMaterial,
    MaterialLibrary,
    MaterialResistance,
    MaterialResistanceValue,
    ModelGlassMaterial,
    PolynomialMaterial,
    SimpleAir,
    Vacuum,
)


class TestMaterial(unittest.TestCase):
    """A class to test the ``Material`` class."""

    def setUp(self):
        """This code is executed again before every test to ensure that the tests are independent."""
        self.vacuum = ModelGlassMaterial("Vacuum")
        self.constant = ModelGlassMaterial("Water", 1.33)
        self.glass = ModelGlassMaterial("FusedSilica", 1.4584, 67.82)

        self.long_wavelength = 656.2725e-9  # red hydrogen line, C
        self.center_wavelength = 587.5618e-9  # yellow helium line, d
        self.short_wavelength = 486.1327e-9  # blue hydrogen line, F

    def test_constant(self):
        npt.assert_array_equal(
            self.vacuum.complex_refractive_index(wavelength=[1e-12, 0.5e-6, 1e-6, np.inf]),
            [1, 1, 1, 1],
        )
        npt.assert_equal(self.vacuum.constringence, np.nan)
        npt.assert_array_equal(
            self.constant.complex_refractive_index(wavelength=[1e-12, 0.5e-6, 1e-6, np.inf]),
            [1.33, 1.33, 1.33, 1.33],
        )
        npt.assert_equal(self.constant.constringence, np.inf)

    def test_realness(self):
        npt.assert_allclose(self.glass.complex_refractive_index(wavelength=np.arange(1, 1000) * 1e-9).imag, 0)
        npt.assert_allclose(
            self.glass.complex_refractive_index(wavelength=np.arange(1, 1000) * 1e-9).real,
            self.glass.refractive_index(wavelength=np.arange(1, 1000) * 1e-9),
        )

    def test_standard_lines(self):
        npt.assert_allclose(
            self.glass.refractive_index(
                wavelength=[self.short_wavelength, self.center_wavelength, self.long_wavelength],
            ),
            [self.glass.refractive_index_F, self.glass.refractive_index_d, self.glass.refractive_index_C],
        )

    def test_constringence(self):
        # TODO: Fix this example
        nC, nd, nF = self.glass.refractive_index_C, self.glass.refractive_index_d, self.glass.refractive_index_F

        npt.assert_equal(self.glass.refractive_index_d, 1.4584)
        npt.assert_equal(self.glass.constringence, (nd - 1) / (nF - nC))
        npt.assert_almost_equal(self.glass.constringence, 67.82, decimal=3)

        npt.assert_array_almost_equal([nC, nd, nF], [1.4562, 1.4584, 1.4630], decimal=4)

    def test_relative_partial_dispersion(self):
        g = self.glass.relative_partial_dispersion_g_F
        npt.assert_equal(np.isfinite(g), True)

    def test_limits(self):
        glass = ModelGlassMaterial("test")
        glass.wavenumber_limits = [2 * np.pi / 500e-9, 2 * np.pi / 1000e-9]
        npt.assert_allclose(glass.wavenumber_limits, sorted([2 * np.pi / 1000e-9, 2 * np.pi / 500e-9]))
        npt.assert_allclose(glass.wavelength_limits, [500e-9, 1000e-9], rtol=1e-9)
        glass.wavelength_limits = [400e-9, 800e-9]
        npt.assert_allclose(glass.wavelength_limits, [400e-9, 800e-9], rtol=1e-9)
        glass.angular_frequency_limits = [2, 4]
        npt.assert_allclose(glass.angular_frequency_limits, [2, 4])

    def test_temperature_pressure(self):
        glass = ModelGlassMaterial("test")
        glass.temperature = 300.0
        glass.pressure = 200000.0
        npt.assert_allclose(glass.temperature, 300.0)
        npt.assert_allclose(glass.pressure, 200000.0)

    def test_refractive_index_by_frequency(self):
        c = 299792458.0
        wavenumber = 2 * np.pi / 587.5618e-9
        angular_frequency = wavenumber * c
        npt.assert_allclose(
            self.glass.refractive_index(angular_frequency=angular_frequency),
            self.glass.refractive_index(wavenumber=wavenumber),
            rtol=1e-9,
        )
        npt.assert_allclose(
            self.glass.refractive_index(wavelength=self.center_wavelength),
            self.glass.refractive_index(wavenumber=wavenumber),
            rtol=1e-9,
        )

    def test_permittivity(self):
        n = self.glass.refractive_index(wavelength=self.center_wavelength)
        eps = self.glass.permittivity(wavelength=self.center_wavelength)
        npt.assert_allclose(eps, n**2, rtol=1e-12)

    def test_transmittance(self):
        t = self.glass.transmittance(thickness=1.0, wavelength=self.center_wavelength)
        npt.assert_allclose(t, 1.0)
        a = self.glass.absorption_coefficient(wavelength=self.center_wavelength)
        npt.assert_allclose(a, 1.0)
        t2 = self.constant.transmittance(thickness=0.5, wavenumber=2 * np.pi / 500e-9)
        npt.assert_allclose(t2, 1.0)
        t3 = self.constant.transmittance(thickness=0.5, angular_frequency=2 * np.pi / 500e-9 * 299792458.0)
        npt.assert_allclose(t3, 1.0)

    def test_repr_hash_eq(self):
        a = ModelGlassMaterial("N-BK7", 1.5168, 64.17)
        b = ModelGlassMaterial("N-BK7", 1.5168, 64.17)
        npt.assert_equal(repr(a).startswith("ModelGlassMaterial("), True)
        npt.assert_equal(a == b, True)
        npt.assert_equal(hash(a) == hash(b), True)
        npt.assert_equal(a == "N-BK7", False)


class TestMaterialResistance(unittest.TestCase):
    def test_resistance_value(self):
        v = MaterialResistanceValue(2)
        npt.assert_equal(float(v), 2.0)
        npt.assert_equal(int(v), 2)
        npt.assert_equal(str(v), "medium-high (2)")
        npt.assert_equal(str(MaterialResistanceValue(1)), "high (1)")
        npt.assert_equal(str(MaterialResistanceValue(5)), "low (5)")
        npt.assert_equal(str(MaterialResistanceValue(3)), "medium-low (3)")
        npt.assert_equal(str(MaterialResistanceValue(4)), "low (4)")
        npt.assert_equal(str(MaterialResistanceValue(1, limits=(1, 5))), "high (1)")
        npt.assert_equal(str(MaterialResistanceValue(51)), "very-low (51)")
        npt.assert_equal(str(MaterialResistanceValue(55)), "too-low (55)")

    def test_resistance(self):
        r = MaterialResistance(
            climate=MaterialResistanceValue(1),
            stain=MaterialResistanceValue(2),
            acid=MaterialResistanceValue(3),
            alkali=MaterialResistanceValue(4),
            phosphate=MaterialResistanceValue(5),
        )
        npt.assert_equal(str(r).startswith("MaterialResistance(climate="), True)
        npt.assert_equal(repr(r).startswith("MaterialResistance(climate=MaterialResistanceValue"), True)


class TestFunctionMaterial(unittest.TestCase):
    def test_permittivity_function(self):
        m = FunctionMaterial(
            "eps",
            permittivity_function=lambda k, t, p: np.full(np.shape(k), 2.25),
        )
        npt.assert_allclose(m.permittivity(wavenumber=[1e6, 2e6]), [2.25, 2.25])
        npt.assert_allclose(m.complex_refractive_index(wavenumber=1e6), 1.5)
        npt.assert_allclose(m.refractive_index(wavelength=500e-9), 1.5)
        npt.assert_allclose(m.refractive_index(angular_frequency=2 * np.pi / 500e-9 * 299792458.0), 1.5)

    def test_no_functions(self):
        with self.assertRaises(ValueError):
            FunctionMaterial("none")

    def test_polynomial_material(self):
        m = PolynomialMaterial(
            "poly",
            refractive_index_model=True,
            factors_um=[1.0],
            exponents=[2],
            poles_um2=[np.nan],
        )
        npt.assert_allclose(m.refractive_index(wavelength=1e-6), 1.0)
        npt.assert_equal(repr(m).startswith("PolynomialMaterial("), True)

    def test_vacuum(self):
        npt.assert_allclose(VACUUM.refractive_index(wavenumber=[1e6, 2e6]), [1, 1])
        npt.assert_allclose(Vacuum().refractive_index(wavelength=500e-9), 1.0)

    def test_simple_air(self):
        air = SimpleAir()
        n = air.refractive_index(wavelength=550e-9)
        npt.assert_allclose(n, 1.00028, atol=1e-4)

    def test_ciddor_air(self):
        air = CiddorAir()
        n = air.refractive_index(wavelength=550e-9)
        npt.assert_allclose(n, 1.00028, atol=1e-4)
        humid = CiddorAir(relative_humidity=0.9, co2_mole_fraction=600e-6)
        n2 = humid.refractive_index(wavelength=550e-9)
        npt.assert_equal(np.isfinite(n2), True)
        npt.assert_equal(n2 != n, True)
        npt.assert_equal(repr(air).startswith("CiddorAir("), True)


class TestMaterialLibrary(unittest.TestCase):
    def setUp(self):
        self.library = MaterialLibrary(
            "test",
            "a test library",
            [
                ModelGlassMaterial("N-BK7", 1.5168, 64.17),
                ModelGlassMaterial("SF11", 1.7847, 25.68),
                ModelGlassMaterial("F2", 1.6186, 36.35),
            ],
        )

    def test_contains(self):
        npt.assert_equal("N-BK7" in self.library, True)
        npt.assert_equal("n-bk7" in self.library, True)
        npt.assert_equal("missing" in self.library, False)
        npt.assert_equal(self.library.materials[0] in self.library, True)
        npt.assert_equal("/^N-/" in self.library, True)
        npt.assert_equal("/^BK/" in self.library, False)
        npt.assert_equal(3.14 in self.library, False)  # ty: ignore[unsupported-operator]

    def test_getitem(self):
        npt.assert_equal(self.library.materials[0].name, "N-BK7")
        npt.assert_equal(self.library["SF11"].name, "SF11")  # ty: ignore[unresolved-attribute]
        npt.assert_equal(self.library["sf11"], None)
        with self.assertRaises(IndexError):
            self.library[17]
        with self.assertRaises(IndexError):
            self.library[-1]
        with self.assertRaises(TypeError):
            self.library[3.14]  # ty: ignore[invalid-argument-type]

    def test_iter(self):
        npt.assert_equal(len(list(self.library)), 3)
        npt.assert_equal([_.name for _ in self.library], ["N-BK7", "SF11", "F2"])

    def test_find_all(self):
        npt.assert_equal(len(self.library.find_all(None)), 3)
        npt.assert_equal([_.name for _ in self.library.find_all("F")], ["SF11", "F2"])
        npt.assert_equal([_.name for _ in self.library.find_all("/^N-/")], ["N-BK7"])
        npt.assert_equal(len(self.library.find_all(None, wavenumber=[2 * np.pi / 500e-9])), 3)
        npt.assert_equal(len(self.library.find_all(None, wavelength=[500e-9, 600e-9])), 3)
        npt.assert_equal(len(self.library.find_all(None, angular_frequency=[1e15])), 3)

    def test_repr(self):
        npt.assert_equal(repr(self.library).startswith("MaterialLibrary(test,"), True)


if __name__ == "__main__":
    unittest.main()
