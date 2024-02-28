"""
Zernike polynomial definition and fitting.

Use :py:func:``fit``(z, y, x, rho, phi, ...) to fit Zernike polynomials to a specified surface `z`. An instance of
:py:class:``Polynomial`` is returned, which is generally a superposition of basis-Zernike-polynomials,
:py:class:``BasisPolynomial``. A :py:obj:``BasisPolynomial(n, m)`` is defined using standard Zernike coefficient or
the radial and azimuthal orders. Conversion functions exist for Noll order. Its arguments can be arrays, and are
broadcasted. The :py:obj:``BasisPolynomial(n, m)(rho, phi)`` object acts as a function in the radial and azimuthal
coordinates. These objects can be used as polar-coordinate functions or as Cartesian functions using the
:py:obj:``BasisPolynomial(n, m).cartesian(y, x)`` property. General :py:class:``Polynomial``s are superpositions of
``BasisPolynomial``s.

Basis Zernike polynomials can be selected using their integer radial and azimuthal ``orders``, standard ``index``,
the ``noll`` index, or the ``fringe`` index. Convert between standard indices and radial+azimuthal order using the
functions :py:func:``index2orders`` and :py:func:``orders2index``. Convert between radial+azimuthal order and Noll
indices using :py:func:``orders2noll`` and :py:func:``noll2orders``. Convert directly between standard and Noll orders
using :py:func:``index2noll`` and :py:func:``noll2index``. Similar functions are provided for Fringe (a.k.a. University
of Arizona) indices.

Commonly used Zernike polynomials have named implementations: :py:func:``piston``, :py:func:``tip``, :py:func:``tilt``,
           :py:func:``oblique_astigmatism``, :py:func:``defocus``, :py:func:``vertical_astigmatism``,
           :py:func:``vertical_trefoil``, :py:func:``vertical_coma``, :py:func:``horizontal_coma``,
           :py:func:``oblique_trefoil``, and :py:func:``primary_spherical``.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from typing import Optional, Callable

from zmxtools.utils.polar import cart2pol
from zmxtools.utils.factorial_fraction import factorial_product_fraction
from zmxtools.utils.array import array_like, asarray, array_type
from zmxtools.utils import script


__all__ = ['index2orders', 'orders2index',
           'noll2orders', 'orders2noll', 'index2noll', 'noll2index',
           'fringe2orders', 'orders2fringe', 'index2fringe', 'fringe2index',
           'BasisPolynomial', 'Polynomial', 'Fit', 'fit',
           'piston',
           'tip', 'tilt',
           'oblique_astigmatism', 'defocus', 'vertical_astigmatism',
           'vertical_trefoil', 'vertical_coma', 'horizontal_coma', 'oblique_trefoil',
           'primary_spherical'
           ]


def index2orders(j_index: array_like) -> (array_type, array_type):
    """
    Converts a Zernike index or indices, js > 0, to a tuple (radial degree m, azimuthal frequency n), for which 0 <= m <= n.
    When multiple values are specified, m and n will have the same shape as the input js.

    The standard OSA/ANSI ordering starts at 0. https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: js = :py:func:``orders2index`(n, m)

    :param j_index: The standard Zernike index, or an ndarray thereof.
    :return: a tuple (n, m) of order subscripts or ndarrays thereof.
    """
    j_index = asarray(j_index, int)

    n = asarray(np.ceil((np.sqrt(9 + 8 * j_index) - 1) / 2) - 1, dtype=int)
    m = 2 * j_index - n * (n + 2)

    n[j_index < 0] = -1  # Mark all indexes less than 0 as invalid

    return n, m


def noll2orders(j_index: array_like) -> (array_type, array_type):
    """
    Converts a Noll index or indices, js > 0, to a tuple (radial degree m, azimuthal frequency n), for which 0 <= m <= n.
    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Noll ordering starts counting at 1, not 0! The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF). J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.

    See also the inverse operation: js = :py:func:``orders2noll``(n, m)

    :param j_index: The standard Zernike index, or an ndarray thereof.
    :return: a tuple (n, m) of order subscripts or ndarrays thereof.
    """
    j_index = asarray(j_index, int)

    n = asarray(np.ceil((np.sqrt(1 + 8 * j_index) - 1) / 2) - 1, dtype=int)
    m_seq = j_index - n * (n + 1) / 2 - 1  # the zero-based sequence number for the real m = 0, 2, -2, 4, -4, 6, -6,... or 1, -1, 3, -3, 6, -6, ... (or the inverse depending on mod(j,2) )
    m = 2 * asarray((m_seq + (1 - np.mod(n, 2))) / 2, dtype=int) + np.mod(n, 2)  # absolute value of real m
    m *= (1 - 2 * np.mod(j_index, 2))  # If j odd, make m negative.

    n[j_index < 1] = -1  # Mark all indexes less than 1 as invalid

    return n, m


def orders2index(n: array_like, m: array_like = 0) -> array_type:
    """
    Converts a Zernike order (radial degree n, azimuthal frequency m), to standard OSA/ANSI j-indices
    When multiple values are specified, j_index will have the same shape as the inputs n and m.

    Invalid indices are marked as -1.

    See also the inverse operation: n, m = :py:func:``index2orders`(j)

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    n = asarray(n, dtype=int)
    m = asarray(m, dtype=int)

    j_index = asarray((n * (n + 2) + m) / 2, dtype=int)

    j_index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return j_index


def orders2noll(n: array_like, m: array_like = 0) -> array_type:
    """
    Converts a Zernike coordinate (radial degree n, azimuthal frequency m), to Noll indexes
    When multiple values are specified, j_index will have the same shape as the inputs n and m.

    The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF). J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.
    Invalid indices are marked as -1.

    See also the inverse operation: n, m = :py:func:``noll2orders`(j_index)

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The Noll Zernike index, or an ndarray thereof.
    """
    n = np.array(n, dtype=int)
    m = np.array(m, dtype=int)

    j_index = asarray(n * (n + 1) / 2, dtype=int)  # number up to n-1
    j_index += np.abs(m) + (m == 0)  # correct number or one too low
    j_index += np.logical_and((m != 0), np.logical_xor(m < 0, np.mod(j_index, 2)))  # make j_index odd if m negative

    j_index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return j_index


def index2noll(j_index: array_like) -> array_type:
    """
    Converts a Zernike index or indices, js > 0, to Noll indexes.
    When multiple values are specified, m and n will have the same shape as the input js.

    The standard OSA/ANSI ordering starts at 0. https://en.wikipedia.org/wiki/Zernike_polynomials
    Invalid indices are marked as -1.

    See also the inverse operation: js = :py:func:``noll2index``(js)

    :param j_index: The standard Zernike index, or an ndarray thereof.
    :return: The Noll Zernike index, or an ndarray thereof.
    """
    return orders2noll(*index2orders(j_index))


def noll2index(j_index: array_like) -> array_type:
    """
    Converts a Noll index or indices, js > 0, to a Zernike index or indices, js > 0.
    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Noll ordering starts counting at 1, not 0! The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF). J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.
    Invalid indices are marked as -1.

    See also the inverse operation: js = :py:func:``index2noll``(js)

    :param j_index: The Noll Zernike index, or an ndarray thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    return orders2index(*noll2orders(j_index))


def orders2fringe(n: array_like, m: array_like = 0) -> array_type:
    """
    Converts a Zernike coordinate (radial degree n, azimuthal frequency m), to fringe (University of Arizona) indexes.
    When multiple values are specified, j_index will have the same shape as the inputs n and m.

    The fringe ordering starts at 1: https://en.wikipedia.org/wiki/Zernike_polynomials
    Invalid indices are marked as -1.

    Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The fringe Zernike index, or an ndarray thereof.
    """
    n = np.array(n, dtype=int)
    m = np.array(m, dtype=int)

    abs_m = np.abs(m)

    j_index = np.array((1 + (n + abs_m) // 2) ** 2 - 2 * abs_m + (m < 0))
    j_index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return j_index


def fringe2orders(j_index: array_like) -> (array_type, array_type):
    """
    Converts a fringe (University of Arizona) index or indices, js > 0, to a tuple (radial degree m, azimuthal
    frequency n), for which 0 <= m <= n. When multiple values are specified, m and n will have the same shape as the
    input js.

    Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.

    See also the inverse operation: js = :py:func:``orders2fringe`(n, m)

    :param j_index: The Fringe Zernike index, or an ndarray thereof.
    :return: a tuple (n, m) of order subscripts or ndarrays thereof.
    """
    j_index = asarray(j_index, int)

    a = 2 * np.floor((j_index - 1) ** 0.5)
    s = (j_index - (a // 2) ** 2 - 1) // 2 * 2
    m_neg = (j_index - (a // 2) ** 2 - 1) % 2

    abs_m = (a - s) // 2
    m = asarray(abs_m * (1 - 2 * m_neg), int)
    n = asarray((a + s) // 2, int)

    n[j_index < 1] = -1  # Mark all indexes less than 1 as invalid

    return n, m


def index2fringe(j_index: array_like) -> array_type:
    """
    Converts a Zernike index or indices, js > 0, to fringe (University of Arizona) indexes.
    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.

    See also the inverse operation: js = :py:func:``fringe2index`(js)

    :param j_index: The standard Zernike index, or an ndarray thereof.
    :return: The Fringe Zernike index, or an ndarray thereof.
    """
    return orders2fringe(*index2orders(j_index))


def fringe2index(j_index: array_like) -> array_type:
    """
    Converts a fringe (University of Arizona) index or indices, js > 0, to a Zernike index or indices, js > 0.
    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.
    https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: j = :py:func:``index2fringe``(j)

    :param j_index: The Fringe Zernike index, or an ndarray thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    return orders2index(*fringe2orders(j_index))


class BasisPolynomial(Callable):  # todo: refactor so that this inherits from Polynomial
    """
    A class representing one of the Zernike basis polynomials, or an array thereof.
    Superpositions of weighted basis polynomials are represented by zernike.Polynomial.
    """
    def __init__(self,
                 index: Optional[array_like] = None,
                 n: Optional[array_like] = None,
                 m: Optional[array_like] = None,
                 odd_and_even: bool = False):
        """
        Constructs one of the Zernike basis polynomial or an array thereof.
        The Zernike basis polynomials form a sqrt(pi) * orthonormal basis on the unit disk, for 2x2-unit square,
        multiply with 4 / pi. The returned Zernike polynomials are themselves functions of polar coordinates (rho=0, phi=0)

        ::

            result = BasisPolynomial(n=n, m=m)

        Returns the Zernike polynomial of radial order n and azimuthal frequency m, where m is between -n and n.

        ::
            result = BasisPolynomial(j)

        Returns the standard OSA/ANSI Zernike polynomial with standard coefficient j_index
        The first of which are: piston,
                                 tilt, tip,
                                 oblique-astigmatism, defocus, vertical-astigmatism,
                                 vertical-trefoil, vertical-coma, horizontal-coma,  horizontal-trefoil,
                                 oblique-trefoil, oblique-quadrafoil, oblique-secondary-astigmatism,
                                 spherical aberration, vertical-secondary-astigmatism vertical-quadrafoil, ...
        where the postscripts indicate the position of the extreme value on the pupil edge.

        When many polynomials need to computed, it will be more efficient to compute multiple polynomials in parallel.
        This function can handle rho and phi matrices and n and m vectors, while the option odd_and_even returns the
        even and odd polynomials as a complex result.

        ::

            result = BasisPolynomial(n=n, m=m, odd_and_even = True)

        For m >= 0, returns the even Zernike polynomial(cos) value as the real part, and the odd polynomial(sin)
        value as the imaginary part. For m < 0, the odd Zernike value is returned as the real part, and the even is
        returned as the imaginary part.

        See also: fit, Fit and Polynomial, index2orders(j), noll2orders(j), orders2index(n, m=0), and orders2noll(n, m=0)

        :param index: (optional) The standard (OSA/ANSI) index of the polynomial. This can a non-negative integer or an nd-array of such integers.
        :param n: (optional) The radial order of the polynomial. This can a non-negative integer or an nd-array of such integers.
        :param m: (optional) The azimuthal frequency of the polynomial. This can a integer <= n or an nd-array of such integers.
        :param odd_and_even: A boolean to indicate if the odd or even counterpart should also be returned. When set to true,
            the imaginary parts of the result contain the counterpart of the requested polynomial (default: False).
        """
        if index is not None:
            if n is None and m is None:
                n, m = index2orders(index)
            else:
                raise ValueError('When the j-index of the Zernike basis polynomial is specified, ' +
                                 'neither order n, nor order m, should be specified.')
        else:
            if n is None or m is None:
                raise ValueError('When the j-index of the Zernike basis polynomial is not specified, ' +
                                 'both order n, and order m, should be specified.')

        self.__n = None
        self.n = asarray(n, int)
        self.__m = None
        self.m = asarray(m, int)

        self.odd_and_even = odd_and_even

    def __call__(self, rho: array_like = 0, phi: array_like = 0) -> array_type:
        """
        Returns the Zernike polynomial of order (n, m) evaluated in polar coordinates at rho and phi.
        The result is represented as a numpy ndarray of dimensions equal, or broadcastable, to the shape of rho
        (and theta), or higher dimensions when n and m are also vectors or arrays.
        The arrays: n, m, j, rho, and phi must be broadcastable.

        :param rho: The radian coordinate. When negative, phi is changed by pi.
            This can be a single number or an nd-array with shape that is broadcastable with the orders n and m of the polynomial.
        :param phi: The azimuthal coordinate [-pi, pi). This can be a single number or an nd-array with shape that is
            broadcastable with the orders n and m of the polynomial.
        :return: A numpy ndarray of dimensions equal to the shape of rho (and phi),
            or higher dimensions when n and m are also vectors or arrays.
        """
        return self.polar(rho=rho, phi=phi)

    def polar(self, rho: array_like = 0, phi: array_like = 0) -> array_type:
        """
        Returns the Zernike polynomial of order (n, m) evaluated in polar coordinates at rho and phi.
        The result is represented as a numpy ndarray of dimensions equal, or broadcastable, to the shape of rho
        (and theta), or higher dimensions when n and m are also vectors or arrays.
        The arrays: n, m, j, rho, and phi must be broadcastable.

        :param rho: The radian coordinate. When negative, phi is changed by pi.
            This can be a single number or an nd-array with shape that is broadcastable with the orders n and m of the polynomial.
        :param phi: The azimuthal coordinate [-pi, pi). This can be a single number or an nd-array with shape that is
            broadcastable with the orders n and m of the polynomial.
        :return: A numpy ndarray of dimensions equal to the shape of rho (and phi),
            or higher dimensions when n and m are also vectors or arrays.
        """
        rho = asarray(rho, float)
        phi = asarray(phi, float)
        # Make orthogonal basis on unit disk (for 2x2 square, set everything outside unit disk to zero and multiply by 4/pi)
        # The norm of each basis vector is sqrt(pi), so that piston(rho, phi) = 1 everywhere.
        normalization = np.sqrt(2 * (self.n + 1) / (1 + (self.m == 0)))
        # Set the real part as requested, the imaginary part will be the odd-counterpart polynomial
        zernike_phase = self.m * (phi + np.pi * (rho < 0)) + (self.m < 0) * np.pi / 2
        if self.odd_and_even:
            zernike_phasor = np.exp(1j * zernike_phase)
        else:
            zernike_phasor = np.cos(zernike_phase)
        result = normalization * self.__polynomial_r(np.abs(rho)) * zernike_phasor

        return result

    @property
    def n(self) -> array_type:
        """
        Get the radial order of the Zernike polynomial.
        """
        return self.__n

    @n.setter
    def n(self, new_radial_order: array_like):
        """
        Set the radial order of the Zernike polynomial.
        """
        self.__n = asarray(new_radial_order, int)

    @property
    def m(self) -> array_type:
        """
        Get the azimuthal order of the Zernike polynomial.
        """
        return self.__m

    @m.setter
    def m(self, new_azimuthal_order: array_like):
        """
        Set the azimuthal order of the Zernike polynomial.
        """
        self.__m = asarray(new_azimuthal_order, int)

    @property
    def index(self) -> array_type:
        """
        Get the standard OSA/ANSI index of the Zernike polynomial.
        """
        return orders2index(n=self.n, m=self.m)

    @index.setter
    def index(self, new_index: array_like):
        """
        Set the standard OSA/ANSI index of the Zernike polynomial.
        """
        self.n, self.m = index2orders(new_index)

    def cartesian(self, y: array_like, x: array_like) -> array_type:
        return self.polar(rho=np.sqrt(y**2 + x**2), phi=np.arctan2(y, x))

    def polar_gradient(self, rho: array_like = 0, phi: array_like = 0) -> array_type:
        """
        Returns the gradient from polar coordinates.
        The first (left-most) dimension has size 2 with the partial derivatives in the order [d_rho, d_phi].
        """
        rho = asarray(rho, float)
        phi = asarray(phi, float)
        # Make orthogonal basis on unit disk (for 2x2 square, set everything outside unit disk to zero and multiply by 4/pi)
        # The norm of each basis vector is sqrt(pi), so that piston(rho, phi) = 1 everywhere.
        normalization = np.sqrt(2 * (self.n + 1) / (1 + (self.m == 0)))
        # Set the real part as requested, the imaginary part will be the odd-counterpart polynomial
        zernike_phase = self.m * (phi + np.pi * (rho < 0)) + (self.m < 0) * np.pi / 2
        if self.odd_and_even:
            zernike_phasor = np.exp(1j * zernike_phase)
            d_zernike_phasor = zernike_phasor * 1j * zernike_phase * self.m
        else:
            zernike_phasor = np.cos(zernike_phase)
            d_zernike_phasor = - np.sin(zernike_phase) * zernike_phase * self.m
        result = normalization * self.__polynomial_r(np.abs(rho)) * zernike_phasor

        rho2 = rho ** 2
        rho2m1 = rho2 - 1

        dZdrho = (
                     (2 * self.n * self.m * rho2m1 + (self.n - self.m) * (self.m + self.n * (2 * rho2 - 1))) * self.__polynomial_r(np.abs(rho))
                     - (self.n + self.m) * (self.n - self.m) * self.__polynomial_r(np.abs(rho))
                  ) / (2 * self.n * rho * rho2m1) * normalization * zernike_phasor
        dZdphi = normalization * self.__polynomial_r(np.abs(rho)) * d_zernike_phasor

        return np.stack([dZdrho, dZdphi])


    def cartesian_gradient(self, y: array_like, x: array_like) -> array_type:
        """Returns the gradient from Cartesian coordinates."""
        return self.polar_gradient(rho=np.sqrt(y**2 + x**2), phi=np.arctan2(y, x))

    def __polynomial_r(self, rho: array_like=0):
        """
        Calculate the radial polynomial component, for all rho in a matrix
        prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case the difference n - m is odd.

        :param rho: An nd-array with the radial distances. This array must have singleton. Non negativeness is enforced.
        :return: The polynomial values in an nd-array of the same shape as rho_i, but broadcasted over the dimensions
        of n and m.
        """
        rho = np.abs(asarray(rho, float))
        if rho.dtype == int:
            rho = rho.astype(float)

        # Complete the shapes of ns and ms
        n, m = np.broadcast_arrays(self.n, np.abs(self.m))  # Make also sure that m is non-negative
        return self.__polynomial_r_static(n, m, rho)

    def __d_polynomial_r(self, rho: array_like=0):
        """
        Calculate the derivative of the radial polynomial component, for all rho in a matrix
        prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case the
        difference n - m is odd.

        :param rho: An nd-array with the radial distances. This array must have singleton. Non negativeness is enforced.

        :return: The polynomial derivatives in an nd-array of the same shape as rho_i, but broadcasted over the dimensions
        of n and m.
        """
        rho = np.abs(asarray(rho, float))
        if rho.dtype == int:
            rho = rho.astype(float)

        # Complete the shapes of ns and ms
        n, m = np.broadcast_arrays(self.n, np.abs(self.m))  # Make also sure that m is non-negative
        return self.__d_polynomial_r_static(n, m, rho)

    @staticmethod
    # TODO: may need caching
    def __polynomial_r_static(n: array_like, m: array_like, rho: array_like = 0.0):
        """
        Calculate the radial polynomial, for all rho in a matrix
        prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case the difference n - m is odd.

        :param n: A non-negative integer or array_like indicating the radial order.
        :param m: An integer or array_like indicating the azimuthal order.
        :param rho: An nd-array with the radial distances. Non-negativeness is enforced.
        :return: The polynomial values in an nd-array of the same shape as rho_i, but broadcasted over the dimensions
        of n and m.
        """
        n_m_dim = n.ndim

        # Expand the output to the shape of that of rho_i broadcasted with n and m
        if rho.ndim < 1:
            rho = rho[..., np.newaxis]
        output_shape = (*rho.shape[:rho.ndim-n_m_dim], *np.maximum(np.array(n.shape), np.array(rho.shape[rho.ndim-n_m_dim:])))
        calculation_shape = (*output_shape[:len(output_shape)-n_m_dim], np.prod(output_shape[len(output_shape)-n_m_dim:], dtype=int))
        rho = np.broadcast_to(rho, shape=output_shape)

        # Start with the first n_m_dim dimensions flattened
        result = np.zeros(shape=calculation_shape)
        rho = np.reshape(rho, newshape=calculation_shape)
        for idx in range(n.size):
            n_i = n.ravel()[idx]
            m_i = m.ravel()[idx]
            rho_i = rho[..., idx]
            if np.mod(n_i - m_i, 2) == 0:  # Skip odd differences, for these the result is zero.
                rho_pow = rho_i**m_i
                rho_sqd = rho_i**2

                coefficients = np.arange((n_i - m_i) / 2, -1, -1, dtype=int)
                for c_idx, c in enumerate(coefficients):
                    # For speedup: rho_pow = rho_i**(n_i-2*coefficients)
                    if c_idx > 0:
                        rho_pow *= rho_sqd  # note the coefficients are in reversed order

                    sub_result_weight = ((-1.0)**c) * factorial_product_fraction(n_i - c, (c, (n_i + m_i) / 2 - c,
                                                                                           (n_i - m_i) / 2 - c))
                    result[..., idx] += sub_result_weight * rho_pow

        return result.reshape(output_shape)

    @property
    def name(self) -> str:
        def radial_multiplicity(_: int) -> str:
            prefixes = ['0-', 'prim', 'second', 'terti', 'quatern', 'quint', 'sext', 'sept', 'oct']
            if _ < len(prefixes):
                result = prefixes[_]
            else:
                result = f'{_}-'
            return result + 'ary '

        def azimulthal_multiplicity(m: int) -> str:
            _ = abs(m)
            special_names = ['spherical', 'coma', 'astigmatism']
            if _ < len(special_names):
                return special_names[_]
            else:
                prefixes = ['0-', '1-', '2-', 'tre', 'quadra', 'penta', 'hexa', 'hepta', 'octa', 'nona', 'deca']
                if _ < len(prefixes):
                    prefix = prefixes[_]
                else:
                    prefix = f'{_}-'
                return prefix + 'foil'

        if self.m == 0:
            if self.n == 0:
                name = 'piston'
            elif self.n == 2:
                name = 'defocus'
            else:
                name = radial_multiplicity(self.n // 2 - 1) + azimulthal_multiplicity(self.m)  # Start counting from spherical
        elif abs(self.m) == 1:
            if self.n == 1:
                name = 'tilt' if self.m < 0 else 'tip'
            else:
                name = 'vertical ' if self.m < 0 else 'horizontal '
                if self.n > 3:
                    name += radial_multiplicity((self.n - abs(self.m)) // 2)
                name += azimulthal_multiplicity(self.m)
        else:
            if abs(self.m) % 2 == 0:
                name = 'oblique ' if self.m < 0 else 'vertical '
            else:
                name = 'vertical ' if self.m < 0 else ('horizontal ' if self.n > 3 else 'oblique ')
            if self.n > abs(self.m):
                name += radial_multiplicity(1 + (self.n - abs(self.m)) // 2)
            name += azimulthal_multiplicity(self.m)

        return name

    def __str__(self) -> str:
        return f'Z{script.sub(self.n)}{script.super(self.m)}'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.index}) = {self.name}"


class Polynomial(Callable):
    def __init__(self, coefficients: array_like = tuple[float](),
                 indices: array_like = tuple[int]()):
        """
        Construct an object that represents superpositions of basis-Zernike polynomials.

        :param coefficients: The coefficients of the polynomials.
        :param indices: The standard indices of the polynomials (Default: all starting from 0).
        """
        self.__coefficients = None
        self.__indices = None
        self.__polynomials: Optional[BasisPolynomial] = None

        self.coefficients = coefficients
        self.indices = indices

    @property
    def indices(self) -> array_like:
        """The standard ISO/ANSI indices of the basis polynomials."""
        if self.__indices.size >= self.coefficients.size:
            return self.__indices[:self.coefficients.size]

        # Pad to right length
        indices = np.arange(len(self.coefficients)).astype(int)
        indices[:len(self.__indices)] = self.__indices
        return indices

    @indices.setter
    def indices(self, new_indices: array_like):
        self.__polynomials = None
        if not np.array_equal(self.__indices, new_indices):
            self.__polynomials = None
            self.__indices = np.array(new_indices)

    @property
    def coefficients(self) -> array_type:
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, new_coefficients: array_like):
        if not np.array_equal(self.__coefficients, new_coefficients):
            self.__polynomials = None
            self.__coefficients = np.array(new_coefficients)

    @property
    def polynomials(self) -> BasisPolynomial:
        if self.__polynomials is None:
            self.__polynomials = BasisPolynomial(index=self.indices)
        return self.__polynomials

    @property
    def n(self) -> array_type:
        """
        Get the radial order of the Zernike polynomial.
        """
        return index2orders(self.indices)[0]

    @n.setter
    def n(self, new_radial_order: array_like):
        """
        Set the radial order of the Zernike polynomial.
        """
        n, m = index2orders(self.indices)
        self.indices = orders2index(new_radial_order, m)

    @property
    def m(self) -> array_type:
        """
        Get the azimuthal order of the Zernike polynomial.
        """
        return index2orders(self.indices)[1]

    @m.setter
    def m(self, new_azimuthal_order: array_like):
        """
        Set the azimuthal order of the Zernike polynomial.
        """
        n, m = index2orders(self.indices)
        self.indices = orders2index(n, new_azimuthal_order)

    @property
    def order(self) -> int:
        return 1 + np.amax(self.indices)

    def complex(self, z: array_like = 0.0) -> array_type:
        """
        Evaluate this polynomial in the complex plane, where the real and imaginary parts of the complex numbers on the
        Argand diagram are interpreted as Cartesian `x` and `y`-coordinates.

        :param z: Complex numbers, where the real part is the horizontal, `x`, coordinate in the pupil with radius 1,
            and their imaginary part is the normalized `y`-coordinate (vertical).

        :return: The value of the polynomial at the specified coordinates.
        """
        return self.cartesian(y=asarray(z, complex).imag, x=asarray(z, complex).real)

    def cartesian(self, y: array_like = 0, x: array_like = 0
                  ) -> array_type:
        """
        Evaluate this polynomial at Cartesian coordinates.

        :param y: The vertical coordinate between -1 and 1.
        :param x: The horizontal coordinate between -1 and 1.

        :return: The value of the polynomial at the specified coordinates.
        """
        return self.polar(rho=np.sqrt(y**2 + x**2), phi=np.arctan2(y, x))

    def polar(self, rho: array_like = 0, phi: array_like = 0
              ) -> array_type:
        """
        Evaluate this polynomial at polar coordinates.

        :param rho: The radial coordinate between 0 and 1. Negative values are interpreted as a phase change of pi.
        :param phi: The angular coordinate in radians.
        :return: The value of the polynomial at the specified coordinates.
        """
        # Add one axis to the left for broadcasting over the self.__polynomials representation
        rho = np.array(rho)[..., np.newaxis]
        phi = np.array(phi)[..., np.newaxis]

        mat = self.polynomials(rho, phi)
        result = mat @ self.coefficients
        return result

    def __call__(self,
                 rho: Optional[array_like] = None,
                 phi: Optional[array_like] = None,
                 y: Optional[array_like] = None,
                 x: Optional[array_like] = None
                 ) -> array_type:
        """
        Evaluate this polynomial at polar, cartesian, or complex coordinates.
        When neither rho or phi are specified, Carthesian coordinates are assumed.
        When rho is specified, but not phi, rho is assumed to contain complex coordinates.
        When both rho and phi are specified, polar coordinates are assumed.

        :param rho: The radial coordinate between 0 and 1, or complex Argand-diagram coordinates.
        :param phi: The angular coordinate in radians.
        :param y: The vertical coordinate between -1 and 1.
        :param x: The horizontal coordinate between -1 and 1.

        :return: The value of the polynomial at the specified coordinates.
        """
        if rho is not None:
            if phi is not None:
                return self.polar(rho=rho, phi=phi)
            else:
                return self.complex(z=rho)
        else:
            return self.cartesian(y=y, x=x)

    def __add__(self, other: array_like | Polynomial) -> Polynomial:
        """
        Return a polynomial that represents that sum of this and another polynomial. If the right-hand side is a
        number, it is interpreted as piston.
        """
        if not isinstance(other, Polynomial):
            other = other * piston
        new_coefficients = defaultdict(int)
        for _, c in zip(self.indices, self.coefficients):
            new_coefficients[_] = c
        for _, c in zip(other.indices, other.coefficients):
            new_coefficients[_] = new_coefficients[_] + c
        combined_indices = sorted(new_coefficients.keys())
        combined_coefficients = [new_coefficients[_] for _ in combined_indices]
        return Polynomial(coefficients=combined_coefficients, indices=combined_indices)

    def __radd__(self, other: array_like) -> Polynomial:
        """
        Return a polynomial that represents that sum of this and another polynomial. If the left-hand side is a
        number, it is interpreted as piston.
        """
        return other + self

    def __neg__(self, other: array_like | Polynomial) -> Polynomial:
        """Return the negated Polynomial, with the sign of all coefficients changed."""
        return Polynomial(coefficients=-self.coefficients, indices=self.indices)

    def __sub__(self, other: array_like | Polynomial) -> Polynomial:
        """
        Return a polynomial that represents that difference of this and another polynomial. If the right-hand side is a
        number, it is interpreted as piston.
        """
        return self + (-other)

    def __rsub__(self, other: array_like) -> Polynomial:
        """
        Return a polynomial that represents that difference of this and another polynomial. If the left-hand side is a
        number, it is interpreted as piston.
        """
        return (-self) + other

    def __mul__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one multiplied by a scalar factor."""
        return Polynomial(coefficients=self.coefficients * other)

    def __rmul__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one multiplied by a scalar factor."""
        return Polynomial(coefficients=self.coefficients * other)

    def __div__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return Polynomial(coefficients=self.coefficients / other)

    def __truediv__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return self / other

    def __rdiv__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return self / other

    def __imul__(self, other: float):
        """In-place multiply (*=) this Polynomial by a scalar constant."""
        self.coefficients *= other
        return self

    def __idiv__(self, other: float):
        """In-place divide (/=) this Polynomial by a scalar constant."""
        self.coefficients /= other
        return self

    def __str__(self) -> str:
        descriptions = list[str]()
        for c, _ in zip(self.coefficients, self.indices):
            c_str = str(c) if c != 1.0 else ""
            if len(descriptions) > 0 and not c_str.startswith("-"):
                descriptions.append("+")
            descriptions.append(c_str)
            n, m = index2orders(_)
            descriptions.append(f'Z{script.sub(n)}{script.super(m)}')

        return ''.join(descriptions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(coefficients={self.coefficients}, indices={self.indices})"


# Some definitions for convenience. For more names, check the BasisPolynomial.name property.
piston = BasisPolynomial(n=0, m=0)

tip = BasisPolynomial(n=1, m=-1)
tilt = BasisPolynomial(n=1, m=1)

oblique_astigmatism = BasisPolynomial(n=2, m=-2)
defocus = BasisPolynomial(n=2, m=0)
vertical_astigmatism = BasisPolynomial(n=2, m=-2)

vertical_trefoil = BasisPolynomial(n=3, m=-3)
vertical_coma = BasisPolynomial(n=3, m=-1)
horizontal_coma = BasisPolynomial(n=3, m=1)
oblique_trefoil = BasisPolynomial(n=3, m=3)

primary_spherical = BasisPolynomial(n=4, m=0)


class Fit(Polynomial):
    """
    A class to fits Zernike polynomials up to the given order.
    """
    def __init__(self,
                 z: array_like,
                 y: Optional[array_like] = None,
                 x: Optional[array_like] = None,
                 rho: Optional[array_like] = None,
                 phi: Optional[array_like] = None,
                 weights: array_like = 1.0,
                 order: array_like = 15):
        """
        Construct an object to fit Zernike polynomials up to the given order.

        :param z: An nd-array of at least dimension 2, the right-most dimensions are indexed by
            either x and y or by rho and phi.
        :param y: The second Cartesian coordinate. Default: covering the range [-1, 1)
        :param x: The first Cartesian coordinate. Default: covering the range [-1, 1)
        :param rho: Alternative radial coordinate when not using Cartesian coordinates.
        :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates.
        :param weights: An nd-array with per-value weights for the fit.
            Default: None = uniform weighting on the unit disk.
        :param order: The number of polynomial terms to consider.
        """
        super().__init__(coefficients=[])

        if np.isscalar(z):
            z = [z]
        z = asarray(z, complex)

        cartesian = rho is None and phi is None
        if cartesian:
            if x is None and y is None:
                x, y = np.linspace(-1, 1, z.shape[-2]), np.linspace(-1, 1, z.shape[-1])

            rho, phi = cart2pol(y, x)

        calc_broadcast = np.broadcast(rho, phi)
        if weights is None:
            # Polar     => proportional to rho in unit disk, 0 outside
            # Cartesian => uniform in unit disk, 0 outside
            inside = np.broadcast_to(rho <= 1, shape=calc_broadcast.shape)
            weights = inside.astype(float)
        else:
            weights = np.broadcast_to(weights, shape=calc_broadcast.shape).astype(float)

        z = np.reshape(z, newshape=(*z.shape[:-2], -1))  # collapse the final (right-most) two dimensions.

        self.__rho = rho
        self.__phi = phi
        self.__z = z
        self.__weights = weights
        self.order = order

        # Lazily calculated
        self.__error = None

    @property
    def order(self) -> int:
        return self.coefficients.size

    @order.setter
    def order(self, new_order: int):
        """
        Sets the order and fits Zernike basis polynomials up to it.
        :param new_order: The number of polynomials to fit.
        """
        self.coefficients = np.zeros(new_order)  # Also determine the polynomials in the super class
        self.coefficients = self.contravariant

    @property
    def contravariant(self) -> array_type:
        basis_vectors = self.polynomials(self.__rho[..., np.newaxis], self.__phi[..., np.newaxis]) \
                        * self.__weights[..., np.newaxis]
        basis_vectors = basis_vectors.reshape(-1, self.order)

        # log.info('Fitting...')
        coefficients, residuals, rank, s = np.linalg.lstsq(basis_vectors, self.__z, rcond=None)
        # log.info(f'residuals={residuals}, rank={rank}, s={s}')

        return coefficients  # Set the coefficients of the underlying Polynomial

    @property
    def covariant(self) -> array_type:
        coefficients = np.zeros(shape=(*self.__z.shape[:-2], self.order))
        for idx in range(self.order):
            basis_vector = (BasisPolynomial(idx)(self.__rho, self.__phi) * self.__weights).ravel()
            coefficients[..., idx] = basis_vector[np.newaxis, :] @ self.__z[..., np.newaxis]
            # log.debug(f"{idx}, {BasisPolynomial(idx).name}: {coefficients[idx]}")
        return coefficients  # Set the coefficients of the underlying Polynomial

    @property
    def error(self) -> float:
        """
        The root-mean-square (RMS) fitting error between `f` and `z`.
        ||z - f|| / sqrt(n), where `n` is the number of sample points.
        """
        if self.__error is None:
            self.__error = np.linalg.norm(self(rho=self.__rho, phi=self.__phi) - self.__z) / np.sqrt(self.__z.size)
        return self.__error

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(coefficients={self.coefficients})"

    # def __mul__(self, other: float):
    #     return Fit(z=self.__z*other, rho=self.__rho, phi=self.__phi, weights=self.__weights, order=self.order)
    #
    # def __rdiv__(self, other):
    #     return Fit(z=self.__z/other, rho=self.__rho, phi=self.__phi, weights=self.__weights, order=self.order)


def fit(z: array_like,
        y: Optional[array_like] = None,
        x: Optional[array_like] = None,
        rho: Optional[array_like] = None,
        phi: Optional[array_like] = None,
        weights: array_like = 1.0,
        order: array_like = 15) -> Fit:
    """
    Fits Zernike polynomial up to the given order and returns a Fit object with the coefficients,
     the polynomial, and the fitting error.

    :param z: An nd-array of at least dimension 2, the right-most dimensions are indexed by
    either x and y or by rho and phi.
    :param y: The second Cartesian coordinate. Default: covering the range [-1, 1)
    :param x: The first Cartesian coordinate. Default: covering the range [-1, 1)
    :param rho: Alternative radial coordinate when not using Cartesian coordinates.
    :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates.
    :param weights: An nd-array with per-value coefficients for the fit.
    Default: None = uniform weighting on the unit disk, weighted by rho in case of polar coordinate specification.
    :param order: The number of polynomial terms to consider.
    :return: The Fit object representing the polynomial.
    """
    return Fit(z=z, y=y, x=x, rho=rho, phi=phi, weights=weights, order=order)


def fit_coefficients(z: array_like,
                     y: Optional[array_like] = None,
                     x: Optional[array_like] = None,
                     rho: Optional[array_like] = None,
                     phi: Optional[array_like] = None,
                     weights: array_like = 1.0,
                     order: int=15) -> array_type:
    """
    Fits (multiple) Zernike polynomials up to the given order.

    :param z: An nd-array of at least dimension 2, the right-most dimensions are indexed by
    either x and y or by rho and phi.
    :param y: The second Cartesian coordinate. Default: covering the range [-1, 1)
    :param x: The first Cartesian coordinate. Default: covering the range [-1, 1)
    :param rho: Alternative radial coordinate when not using Cartesian coordinates.
    :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates.
    :param weights: An nd-array with per-value coefficients for the fit.
    Default: None = uniform weighting on the unit disk, weighted by rho in case of polar coordinate specification.
    :param order: The number of polynomial terms to consider.
    :return: A vector or nd-array of vectors with the polynomial coefficients.
    """
    cartesian = rho is None and phi is None
    if cartesian:
        if x is None and y is None:
            x, y = np.linspace(-1, 1, z.shape[-2]), np.linspace(-1, 1, z.shape[-1])

        rho, phi = cart2pol(y, x)

    calc_broadcast = np.broadcast(rho, phi)
    if weights is None:
        weights = np.broadcast_to((cartesian + (not cartesian) * rho) * (rho <= 1), shape=calc_broadcast.shape)
    else:
        weights = np.broadcast_to(weights, shape=calc_broadcast.shape)

    weights = weights / np.sum(weights)  # Make sure that the sum of the coefficients is 1

    z = np.reshape(z, newshape=(*z.shape[:-2], -1))  # collapse the final (right-most) two dimensions.
    coefficients = np.zeros(shape=(*z.shape[:-2], order))
    for idx in range(order):
        basis_vector = np.conj(BasisPolynomial(idx)(rho, phi) * weights).flatten()
        coefficients[..., idx] = basis_vector @ z
        # log.debug(f"{idx}, {BasisPolynomial(idx).name}: {coefficients[idx]}")

    return coefficients
