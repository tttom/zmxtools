import unittest
import numpy.testing as npt

from zmxtools.utils.zernike import (index2orders, orders2index,
                                    noll2orders, orders2noll, noll2index, index2noll,
                                    fringe2orders, orders2fringe, fringe2index, index2fringe,
                                    BasisPolynomial, Polynomial, fit)

import numpy as np


class TestPolynomial(unittest.TestCase):
    def test_zernike_index(self):
        piston = BasisPolynomial(0)
        tilt = BasisPolynomial(1)
        tip = BasisPolynomial(2)
        defocus = BasisPolynomial(4)
        npt.assert_equal(piston(0), 1, "Piston fit failed")
        npt.assert_equal(piston(1/2), 1, "Piston fit failed")
        npt.assert_equal(piston(1), 1, "Piston fit failed")
        npt.assert_equal(tip(0.0), 0.0, "Tip fit failed")
        npt.assert_equal(tip(0.5), 1.0, "Tip fit failed")
        npt.assert_equal(tip(1.0), 2.0, "Tip fit failed")
        npt.assert_equal(tip(np.array([0.0, 0.5, 1.0])), np.array([0.0, 1.0, 2.0]), "Tip fit failed")
        npt.assert_almost_equal(tilt(np.array([0.0, 0.5, 1.0])), np.array([0.0, 0.0, 0.0]), 12, "Tilt fit failed")
        npt.assert_equal(tilt(np.array([0.0, 0.5, 1.0]), np.pi/2), np.array([0.0, 1.0, 2.0]), "Rotated tilt fit failed")
        npt.assert_almost_equal(tip(np.array([0.0, 0.5, 1.0]), np.pi/2), np.array([0.0, 0.0, 0.0]), 12, "Rotated tip fit failed")
        npt.assert_almost_equal(
            tilt(np.array([[0.0, 0.5, 1.0]]), [[0], [np.pi/2]]), np.array([[0, 0, 0], [0, 1, 2]]),
            12, "Non|rotated tilt fit failed")
        npt.assert_almost_equal(defocus([[0.0, 0.5, 1.0]], [[0.0], [np.pi/2]]),
                                np.sqrt(3) * np.array([[-1.0, -0.5, 1.0], [-1.0, -0.5, 1.0]]),
                                12, "Defocus fit failed")

    def test_zernike_index_array(self):
        ab4 = BasisPolynomial([0, 2, 1, 4])
        ab22 = BasisPolynomial([[0, 2], [1, 4]])

        npt.assert_equal(ab4(0), np.array([1, 0, 0, -np.sqrt(3)]), "Array of aberrations failed at rho=0")
        npt.assert_array_almost_equal(ab4(1), np.array([1, 2, 0, np.sqrt(3)]), 12, "Array of aberrations failed at rho=1")
        npt.assert_array_almost_equal(ab4([[0], [1]]), np.array([[1, 0, 0, -np.sqrt(3)], [1, 2, 0, np.sqrt(3)]]),
                                      12, "Array of aberrations failed at rho=[0, 1]")

        npt.assert_array_equal(ab22(0), np.array([[1, 0], [0, -np.sqrt(3)]]), "Array of aberrations failed at rho=0")
        npt.assert_array_almost_equal(ab22(1), np.array([[1, 2], [0, np.sqrt(3)]]), 12, "Array of aberrations failed at rho=1")
        npt.assert_array_almost_equal(ab22([[[0]], [[1]]]), np.array([[[1, 0], [0, -np.sqrt(3)]], [[1, 2], [0, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed at rho=[0, 1]")

        npt.assert_array_equal(ab4(0, 0), np.array([1, 0, 0, -np.sqrt(3)]),
                               "Array of aberrations failed at (rho, theta)=(0, 0)")
        npt.assert_array_almost_equal(ab4(1, 0), np.array([1, 2, 0, np.sqrt(3)]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, 0)")
        npt.assert_array_almost_equal(ab4([[0], [1]], [[0], [0]]), np.array([[1, 0, 0, -np.sqrt(3)], [1, 2, 0, np.sqrt(3)]]),
                                      12, "Array of aberrations failed at (rho,theta)=([0, 1],[0, 0])")

        npt.assert_array_equal(ab4(0, np.pi/2), np.array([1, 0, 0, -np.sqrt(3)]),
                               "Array of aberrations failed at (rho, theta)=(0, pi/2)")
        npt.assert_array_almost_equal(ab4(1, np.pi/2), np.array([1, 0, 2, np.sqrt(3)]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, pi/2)")
        npt.assert_array_almost_equal(ab4([[0], [1]], [[np.pi/2], [np.pi/2]]),
                                      np.array([[1, 0, 0, -np.sqrt(3)], [1, 0, 2, np.sqrt(3)]]),
                                      12, "Array of aberrations failed at (rho,theta)=([0, 1],[pi/2, pi/2])")
        npt.assert_array_almost_equal(ab4([[[0]], [[1]]], [[[0], [np.pi/2]]]),
                                      np.array([[[1, 0, 0, -np.sqrt(3)], [1, 0, 0, -np.sqrt(3)]],
                                                [[1, 2, 0, np.sqrt(3)], [1, 0, 2, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed for 2D (rho,theta)")

        npt.assert_array_equal(ab22(0), np.array([[1, 0], [0, -np.sqrt(3)]]), "Array of aberrations failed at rho=0")
        npt.assert_array_almost_equal(ab22(1), np.array([[1, 2], [0, np.sqrt(3)]]), 12, "Array of aberrations failed at rho=1")
        npt.assert_array_almost_equal(ab22([[[0]], [[1]]]), np.array([[[1, 0], [0, -np.sqrt(3)]], [[1, 2], [0, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed at rho=[0, 1]")

        npt.assert_array_equal(ab22(0, 0), np.array([[1, 0], [0, -np.sqrt(3)]]),
                               "Array of aberrations failed at (rho, theta)=(0, 0)")
        npt.assert_array_almost_equal(ab22(1, 0), np.array([[1, 2], [0, np.sqrt(3)]]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, 0)")
        npt.assert_array_almost_equal(ab22([[[0]], [[1]]], 0), np.array([[[1, 0], [0, -np.sqrt(3)]], [[1, 2], [0, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed at (rho, theta)=([0, 1], 0)")
        npt.assert_array_almost_equal(ab22(np.array([0, 1])[:, np.newaxis, np.newaxis, np.newaxis],
                                           np.array([0, np.pi/2])[np.newaxis, :, np.newaxis, np.newaxis]
                                           ),
                                      np.array([[[[1, 0], [0, -np.sqrt(3)]], [[1, 0], [0, -np.sqrt(3)]]],
                                                [[[1, 2], [0, np.sqrt(3)]], [[1, 0], [2, np.sqrt(3)]]]]),
                                      12, "Array of aberrations failed at (rho, theta)=([[0], [1]], [[0, pi/2]])")

        npt.assert_array_equal(ab22(0, np.pi/2), np.array([[1, 0], [0, -np.sqrt(3)]]),
                               "Array of aberrations failed at (rho, theta)=(0, pi/2)")
        npt.assert_array_almost_equal(ab22(1, np.pi/2), np.array([[1, 0], [2, np.sqrt(3)]]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, pi/2)")

    def test_error(self):
        with npt.assert_raises(ValueError):
            BasisPolynomial(2, 0)

    def test_zernike_superposition(self):
        test_args = [(0, 0), (1, 0), (-1, 0), (1, np.pi), (1, np.pi/2), (1, np.pi/4), (0.5, 0.0), (0.5, np.pi/8)]
        test_coefficients = [1, 2, 3, 4]

        s = Polynomial(test_coefficients)
        npt.assert_array_equal(s.coefficients, test_coefficients)
        npt.assert_array_equal(s.indices, np.arange(len(test_coefficients)))
        bs = lambda rho, phi: sum(c * BasisPolynomial(_)(rho, phi) for _, c in enumerate(test_coefficients))

        for args in test_args:
            npt.assert_array_equal(s(*args), bs(*args), err_msg=f"Failed at point (rho, phi) = {args}")

        npt.assert_array_equal(s(-1, 0), bs(1, np.pi),
                               err_msg="Negative radial coordinates should be the same as a pi-phase flip.")
        npt.assert_array_equal(s(-1, np.pi/4), bs(1, np.pi + np.pi/4),
                               err_msg="Negative radial coordinates should be the same as a pi-phase flip.")

        test_indices = noll2index(range(1, 1 + len(test_coefficients)))
        s2 = Polynomial(test_coefficients, indices=test_indices)
        npt.assert_array_equal(s2.coefficients, test_coefficients)
        npt.assert_array_equal(s2.indices, test_indices)
        bs2 = lambda rho, phi: sum(c * BasisPolynomial(test_indices[_])(rho, phi) for _, c in enumerate(test_coefficients))

        for args in test_args:
            npt.assert_array_equal(s2(*args), bs2(*args), err_msg=f"Failed at point (rho, phi) = {args}")

    def test_zernike_fit_cartesian(self):
        rng = np.linspace(-1, 1, 32)
        y, x = rng[:, np.newaxis], rng[np.newaxis, :]

        piston = BasisPolynomial(0)
        f = fit(z=piston.cartesian(y=y, x=x), y=y, x=x, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([1, 0, 0, 0, 0]), decimal=8)

        defocus = BasisPolynomial(4)
        f = fit(z=defocus.cartesian(y=y, x=x), y=y, x=x, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([0, 0, 0, 0, 1]), decimal=8)

        s = Polynomial([4, 3, 2, 0, 1])
        f = fit(z=s.cartesian(y=y, x=x), y=y, x=x, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([4, 3, 2, 0, 1]), decimal=8)

    def test_zernike_fit_polar(self):
        nb_subdivisions = 32
        rho = np.linspace(0, 1, nb_subdivisions)[:, np.newaxis]
        phi = np.linspace(-np.pi, np.pi, nb_subdivisions + 1, endpoint=False)[np.newaxis, :]

        piston = BasisPolynomial(0)
        f = fit(z=piston(rho=rho, phi=phi), rho=rho, phi=phi, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([1, 0, 0, 0, 0]), decimal=8)

        defocus = BasisPolynomial(4)
        f = fit(z=defocus(rho=rho, phi=phi), rho=rho, phi=phi, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([0, 0, 0, 0, 1]), decimal=8)

        s = Polynomial([4, 3, 2, 0, 1])
        f = fit(z=s(rho=rho, phi=phi), rho=rho, phi=phi, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([4, 3, 2, 0, 1]), decimal=8)


class TestIndexConversion(unittest.TestCase):
    def test_index2orders(self):
        npt.assert_equal(index2orders(0), (0, 0),
                         'Piston not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(1), (1, -1),
                         'Tilt not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(2), (1, 1),
                         'Tip not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(3), (2, -2),
                         'Oblique-astigmatism is not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(4), (2, 0),
                         'Defocus not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(5), (2, 2),
                         'Astigmatism-cartesian not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(6), (3, -3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(7), (3, -1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(8), (3, 1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(9), (3, 3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(12), (4, 0),
                         'Spherical aberration not converted to the correct radial degree and azimuthal frequency.')

    def test_orders2index(self):
        npt.assert_equal(orders2index(0, 0), 0,
                         'Piston not converted to the correct index.')
        npt.assert_equal(orders2index(1, -1), 1,
                         'Tilt not converted to the correct index.')
        npt.assert_equal(orders2index(1, 1), 2,
                         'Tip not converted to the correct index.')
        npt.assert_equal(orders2index(2, -2), 3,
                         'Astigmatism-diag not converted to the correct index.')
        npt.assert_equal(orders2index(2, 0), 4,
                         'Defocus not converted to the correct index.')
        npt.assert_equal(orders2index(2, 2), 5,
                         'Astigmatism-cartesian not converted to the correct index.')
        npt.assert_equal(orders2index(3, -3), 6,
                         'Trefoil- not converted to the correct index.')
        npt.assert_equal(orders2index(3, -1), 7,
                         'Coma not converted to the correct index.')
        npt.assert_equal(orders2index(3, 1), 8,
                         'Coma not converted to the correct index.')
        npt.assert_equal(orders2index(3, 3), 9,
                         'Trefoil+ not converted to the correct index.')
        npt.assert_equal(orders2index(4, 0), 12,
                         'Spherical aberration not converted to the correct index.')

    def test_orders2index2orders_tuple(self):
        npt.assert_equal(orders2index((0, 1, 2), (0, 1, 0)), np.array((0, 2, 4)),
                         'Tuple not converted to the correct index.')
        npt.assert_equal(index2orders((0, 2, 4)), np.array([[0, 1, 2], [0, 1, 0]]),
                         'Tuple not converted to the correct index.')

    def test_orders2index2orders_array(self):
        npt.assert_equal(orders2index(((0, 1), (2, 3)), ((0, 1), (0, -1))), np.array(((0, 2), (4, 7))),
                         'Tuple not converted to the correct index.')
        npt.assert_equal(index2orders([[0, 2], [4, 7]]), (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [0, -1]])),
                         'Nested list of indices not converted to the correct tuple of orders.')


class TestNoll(unittest.TestCase):
    def test_noll2orders(self):
        npt.assert_equal(noll2orders(1), (0, 0),
                         'Piston not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(2), (1, 1),
                         'Tip not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(3), (1, -1),
                         'Tilt not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(4), (2, 0),
                         'Defocus not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(5), (2, -2),
                         'Astigmatism-diag not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(6), (2, 2),
                         'Astigmatism-cartesian not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(7), (3, -1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(8), (3, 1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(9), (3, -3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(10), (3, 3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(11), (4, 0),
                         'Spherical aberration not converted to the correct radial degree and azimuthal frequency.')

    def test_orders2noll(self):
        npt.assert_equal(orders2noll(0, 0), 1,
                         'Piston not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(1, 1), 2,
                         'Tip not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(1, -1), 3,
                         'Tilt not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(2, 0), 4,
                         'Defocus not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(2, -2), 5,
                         'Astigmatism-diag not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(2, 2), 6,
                         'Astigmatism-cartesian not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(3, -1), 7,
                         'Coma not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(3, 1), 8,
                         'Coma not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(3, -3), 9,
                         'Trefoil- not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(3, 3), 10,
                         'Trefoil+ not converted to the correct Noll index.')
        npt.assert_equal(orders2noll(4, 0), 11,
                         'Spherical aberration not converted to the correct Noll index.')

    def test_noll2index(self):
        npt.assert_equal(noll2index(1), 0,
                         'Piston not converted to the correct standard index.')
        npt.assert_equal(noll2index(2), 2,
                         'Tip not converted to the correct standard index.')
        npt.assert_equal(noll2index(3), 1,
                         'Tilt not converted to the correct standard index.')
        npt.assert_equal(noll2index(4), 4,
                         'Defocus not converted to the correct standard index.')
        npt.assert_equal(noll2index(5), 3,
                         'Astigmatism-diag not converted to the correct standard index.')
        npt.assert_equal(noll2index(6), 5,
                         'Astigmatism-cartesian not converted to the correct standard index.')
        npt.assert_equal(noll2index(7), 7,
                         'Coma not converted to the correct standard index.')
        npt.assert_equal(noll2index(8), 8,
                         'Coma not converted to the correct standard index.')
        npt.assert_equal(noll2index(9), 6,
                         'Trefoil not converted to the correct standard index.')
        npt.assert_equal(noll2index(10), 9,
                         'Trefoil not converted to the correct standard index.')
        npt.assert_equal(noll2index(11), 12,
                         'Spherical aberration not converted to the correct standard index.')
        npt.assert_equal(noll2index((1, 2, 3, 4)), np.array((0, 2, 1, 4)),
                         'Tuple of indices not converted to the correct standard indices.')
        npt.assert_equal(noll2index([1, 2, 3, 4]), np.array([0, 2, 1, 4]),
                         'List of indices not converted to the correct standard indices.')

    def test_index2noll(self):
        npt.assert_equal(index2noll(0), 1,
                         'Piston not converted to the correct Noll index.')
        npt.assert_equal(index2noll(2), 2,
                         'Tip not converted to the correct Noll index.')
        npt.assert_equal(index2noll(1), 3,
                         'Tilt not converted to the correct Noll index.')
        npt.assert_equal(index2noll(4), 4,
                         'Defocus not converted to the correct Noll index.')
        npt.assert_equal(index2noll(3), 5,
                         'Astigmatism-diag not converted to the correct Noll index.')
        npt.assert_equal(index2noll(5), 6,
                         'Astigmatism-cartesian not converted to the correct Noll index.')
        npt.assert_equal(index2noll(7), 7,
                         'Coma not converted to the correct Noll index.')
        npt.assert_equal(index2noll(8), 8,
                         'Coma not converted to the correct Noll index.')
        npt.assert_equal(index2noll(6), 9,
                         'Trefoil- not converted to the correct Noll index.')
        npt.assert_equal(index2noll(9), 10,
                         'Trefoil+ not converted to the correct Noll index.')
        npt.assert_equal(index2noll(12), 11,
                         'Spherical aberration not converted to the correct Noll index.')
        npt.assert_equal(index2noll((0, 2, 1, 4)), np.arange(1, 5),
                         'Tuple of indices not converted to the correct Noll indices.')
        npt.assert_equal(index2noll([0, 2, 1, 4]), np.arange(1, 5),
                         'List of indices not converted to the correct Noll indices.')

    def test_orders2noll2orders_tuple(self):
        npt.assert_equal(orders2noll((0, 1, 2), (0, 1, 0)), np.array((1, 2, 4)),
                         'Tuple not converted to the correct Noll index.')
        npt.assert_equal(noll2orders((1, 2, 4)), np.array([[0, 1, 2], [0, 1, 0]]),
                         'Tuple not converted to the correct Noll index.')

    def test_orders2noll2orders_array(self):
        npt.assert_equal(orders2noll(((0, 1), (2, 3)), ((0, 1), (0, -1))), np.array(((1, 2), (4, 7))),
                         'Tuple not converted to the correct Noll indices.')
        npt.assert_equal(noll2orders([[1, 2], [4, 7]]), (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [0, -1]])),
                         'Tuple not converted to the correct Noll indices.')

    def test_index2noll2index(self):
        test = np.arange(1000)
        npt.assert_array_equal(noll2index(index2noll(test)), test, err_msg="index2noll is not the inverse of noll2index.")


class TestFringe(unittest.TestCase):
    def test_fringe2orders(self):
        npt.assert_equal(fringe2orders(1), (0, 0),
                         'Piston not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(2), (1, 1),
                         'Tip not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(3), (1, -1),
                         'Tilt not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(4), (2, 0),
                         'Defocus not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(6), (2, -2),
                         'Astigmatism-diag not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(5), (2, 2),
                         'Astigmatism-cartesian not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(8), (3, -1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(7), (3, 1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(11), (3, -3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(10), (3, 3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(fringe2orders(9), (4, 0),
                         'Spherical aberration not converted to the correct radial degree and azimuthal frequency.')

    def test_orders2fringe(self):
        npt.assert_equal(orders2fringe(0, 0), 1,
                         'Piston not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(1, 1), 2,
                         'Tip not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(1, -1), 3,
                         'Tilt not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(2, 0), 4,
                         'Defocus not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(2, -2), 6,
                         'Astigmatism-diag not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(2, 2), 5,
                         'Astigmatism-cartesian not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(3, -1), 8,
                         'Coma not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(3, 1), 7,
                         'Coma not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(3, -3), 11,
                         'Trefoil- not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(3, 3), 10,
                         'Trefoil+ not converted to the correct Fringe index.')
        npt.assert_equal(orders2fringe(4, 0), 9,
                         'Spherical aberration not converted to the correct Fringe index.')

    def test_fringe2index(self):
        npt.assert_equal(fringe2index(1), 0,
                         'Piston not converted to the correct standard index.')
        npt.assert_equal(fringe2index(2), 2,
                         'Tip not converted to the correct standard index.')
        npt.assert_equal(fringe2index(3), 1,
                         'Tilt not converted to the correct standard index.')
        npt.assert_equal(fringe2index(4), 4,
                         'Defocus not converted to the correct standard index.')
        npt.assert_equal(fringe2index(5), 5,
                         'Astigmatism-diag not converted to the correct standard index.')
        npt.assert_equal(fringe2index(6), 3,
                         'Astigmatism-cartesian not converted to the correct standard index.')
        npt.assert_equal(fringe2index(7), 8,
                         'Coma not converted to the correct standard index.')
        npt.assert_equal(fringe2index(8), 7,
                         'Coma not converted to the correct standard index.')
        npt.assert_equal(fringe2index(9), 12,
                         'Trefoil not converted to the correct standard index.')
        npt.assert_equal(fringe2index(10), 9,
                         'Trefoil not converted to the correct standard index.')
        npt.assert_equal(fringe2index(11), 6,
                         'Spherical aberration not converted to the correct standard index.')
        npt.assert_equal(fringe2index((1, 2, 3, 4)), np.array((0, 2, 1, 4)),
                         'Tuple of indices not converted to the correct standard indices.')
        npt.assert_equal(fringe2index([1, 2, 3, 4]), np.array([0, 2, 1, 4]),
                         'List of indices not converted to the correct standard indices.')

    def test_index2fringe(self):
        npt.assert_equal(index2fringe(0), 1,
                         'Piston not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(2), 2,
                         'Tip not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(1), 3,
                         'Tilt not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(4), 4,  # (n(n+2) + m) / 2
                         'Defocus not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(5), 5,
                         'Astigmatism-diag not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(3), 6,
                         'Astigmatism-cartesian not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(8), 7,
                         'Coma not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(7), 8,
                         'Coma not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(12), 9,
                         'Trefoil- not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(9), 10,
                         'Trefoil+ not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe(6), 11,
                         'Spherical aberration not converted to the correct Fringe index.')
        npt.assert_equal(index2fringe((0, 2, 1, 4)), np.arange(1, 5),
                         'Tuple of indices not converted to the correct Fringe indices.')
        npt.assert_equal(index2fringe([0, 2, 1, 4]), np.arange(1, 5),
                         'List of indices not converted to the correct Fringe indices.')

    def test_orders2fringe2orders_tuple(self):
        npt.assert_equal(orders2fringe((0, 1, 2), (0, 1, 0)), np.array((1, 2, 4)),
                         'Tuple not converted to the correct Fringe indices.')
        npt.assert_equal(fringe2orders((1, 2, 4)), np.array([[0, 1, 2], [0, 1, 0]]),
                         'Tuple not converted to the correct Fringe indices.')

    def test_orders2fringe2orders_array(self):
        npt.assert_equal(orders2fringe(((0, 1), (2, 3)), ((0, 1), (0, 1))), np.array(((1, 2), (4, 7))),
                         'Tuple not converted to the correct Fringe indices.')
        npt.assert_equal(fringe2orders([[1, 2], [4, 7]]), (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [0, 1]])),
                         'Tuple not converted to the correct Fringe indices.')

    def test_index2fringe2index(self):
        test = np.arange(1000)
        npt.assert_array_equal(fringe2index(index2fringe(test)), test, err_msg="index2fringe is not the inverse of fringe2index.")


if __name__ == '__main__':
    unittest.main()
