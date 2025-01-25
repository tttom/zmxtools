import unittest

import numpy.testing as npt
import numpy as np

from zmxtools.optical_design.material import ModelGlassMaterial


class TestMaterial(unittest.TestCase):
    def setUp(self):
        self.vacuum = ModelGlassMaterial('Vacuum')
        self.constant = ModelGlassMaterial('Water', 1.33)
        self.glass = ModelGlassMaterial('BK7', 1.52, 64)

        self.long_wavelength = 656.2725e-9  # red hydrogen line, C
        self.center_wavelength = 587.5618e-9  # yellow helium line, d
        self.short_wavelength = 486.1327e-9  # blue hydrogen line, F

    def test_constructors(self):
        npt.assert_array_equal(self.vacuum.complex_refractive_index(wavelength=[0, 0.5e-6, 1e-6]),
                               [1, 1, 1])
        npt.assert_array_equal(self.constant.complex_refractive_index(wavelength=[0, 0.5e-6, 1e-6]),
                               [1.33, 1.33, 1.33])
        nF, nd, nC = self.glass.complex_refractive_index(
            wavelength=[self.long_wavelength, self.center_wavelength, self.short_wavelength],
        )
        npt.assert_array_equal([nd, (nd - 1) / (nC - nF)], [1.52, 64])


if __name__ == '__main__':
    unittest.main()
