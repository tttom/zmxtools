import unittest

import numpy.testing as npt
import numpy as np

from zmxtools.optical_design.material import ModelGlassMaterial


class TestMaterial(unittest.TestCase):
    def setUp(self):
        self.vacuum = ModelGlassMaterial('Vacuum')
        self.constant = ModelGlassMaterial('Water', 1.33)
        self.glass = ModelGlassMaterial('FusedSilica', 1.45846, 67.82)

        self.long_wavelength = 656.2725e-9  # red hydrogen line, C
        self.center_wavelength = 587.5618e-9  # yellow helium line, d
        self.short_wavelength = 486.1327e-9  # blue hydrogen line, F

    def test_constant(self):
        npt.assert_array_equal(self.vacuum.complex_refractive_index(wavelength=[0, 0.5e-6, 1e-6, np.inf]),
                               [1, 1, 1, 1])
        npt.assert_equal(self.vacuum.constringence, np.nan)
        npt.assert_array_equal(self.constant.complex_refractive_index(wavelength=[0, 0.5e-6, 1e-6, np.inf]),
                               [1.33, 1.33, 1.33, 1.33])
        npt.assert_equal(self.constant.constringence, np.inf)

    def test_realness(self):
        npt.assert_allclose(self.glass.complex_refractive_index(wavelength=np.arange(1, 1000) * 1e-9).imag, 0)
        npt.assert_allclose(self.glass.complex_refractive_index(wavelength=np.arange(1, 1000) * 1e-9).real,
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

        npt.assert_equal(self.glass.refractive_index_d, 1.45846)
        npt.assert_equal(self.glass.constringence, (nd - 1) / (nF - nC))
        npt.assert_equal(self.glass.constringence, 67.82)

        # npt.assert_array_almost_equal([nC, nd, nF, self.glass.constringence], [1.4565, 1.4585, 1.4627, 67.82], decimal=4)


if __name__ == '__main__':
    unittest.main()
