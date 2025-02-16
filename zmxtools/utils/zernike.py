"""
Zernike polynomial definition and fitting.

Use :py:func:``fit``(z, y, x, rho, phi, ...) to fit Zernike polynomials to a specified surface `z`. An instance of
:py:class:``Polynomial`` is returned, which is generally a superposition of basis-Zernike-polynomials,
:py:class:``PolynomialBasis``. A :py:obj:``PolynomialBasis(n, m)`` is defined using standard Zernike coefficient or
the radial and azimuthal orders. Conversion functions exist for Noll order. Its arguments can be arrays, and are
broadcasted. The :py:obj:``PolynomialBasis(n, m)(rho, phi)`` object acts as a function in the radial and azimuthal
coordinates. These objects can be used as polar-coordinate functions or as Cartesian functions using the
:py:obj:``PolynomialBasis(n, m).cartesian(y, x)`` property. General :py:class:``Polynomial``s are superpositions of
``PolynomialBasis``s.

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
from math import prod
from typing import Dict, TypeVar, assert_type

import numpy as np

from zmxtools.utils import script
from zmxtools.utils.array import (FLOAT_TYPE, INT_TYPE, NP_COMPLEX_TYPE, NP_FLOAT_TYPE, NP_INT_TYPE,
                                  array_like, array_type, asarray)
from zmxtools.utils.factorial_fraction import factorial_product_fraction
from zmxtools.utils.polar import cart2pol
from zmxtools.utils import log

log = log.getChild(__name__)

__all__ = ['index2orders', 'orders2index',
           'noll2orders', 'orders2noll', 'index2noll', 'noll2index',
           'fringe2orders', 'orders2fringe', 'index2fringe', 'fringe2index',
           'PolynomialBasis', 'Polynomial', 'Fit', 'fit',
           'piston',
           'tip', 'tilt',
           'oblique_astigmatism', 'defocus', 'vertical_astigmatism',
           'vertical_trefoil', 'vertical_coma', 'horizontal_coma', 'oblique_trefoil',
           'primary_spherical',
           ]


def index2orders(index: array_like[NP_INT_TYPE]) -> tuple[array_type[NP_INT_TYPE], array_type[NP_INT_TYPE]]:
    """
    Converts Zernike indices, js > 0, to a tuple (radial degree m, azimuthal frequency n), for which 0 <= m <= n.

    When multiple values are specified, m and n will have the same shape as the input js.

    The standard OSA/ANSI ordering starts at 0. https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: js = :py:func:``orders2index`(n, m)

    :param index: The standard Zernike index, j, or an ndarray thereof.
    :return: a tuple (n, m) of order subscripts or ndarrays thereof.
    """
    index = asarray(index, int)

    n = asarray(np.ceil((np.sqrt(9 + 8 * index) - 1) / 2) - 1, int)
    m = 2 * index - n * (n + 2)

    n[index < 0] = -1  # Mark all indexes less than 0 as invalid

    return n, m


def noll2orders(index: array_like[NP_INT_TYPE]) -> tuple[array_type[NP_INT_TYPE], array_type[NP_INT_TYPE]]:
    """
    Converts Noll indices, js > 0, to a tuple (radial degree m, azimuthal frequency n), for which 0 <= m <= n.

    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Noll ordering starts counting at 1, not 0! The ordering is described here:
        Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF).
        J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.

    See also the inverse operation: js = :py:func:``orders2noll``(n, m)

    :param index: The Noll index, or an ndarray thereof.
    :return: a tuple (n, m) of order subscripts or ndarrays thereof.

    """
    index = asarray(index, int)

    n = asarray(np.ceil((np.sqrt(1 + 8 * index) - 1) / 2) - 1, dtype=int)
    m_seq = index - n * (n + 1) / 2 - 1  # the zero-based sequence number for the real m = 0, 2, -2, 4, -4, 6, -6,...
    # or 1, -1, 3, -3, 6, -6, ... (or the inverse depending on mod(j,2) )
    m = 2 * asarray((m_seq + (1 - np.mod(n, 2))) / 2, dtype=int) + np.mod(n, 2)  # absolute value of real m
    m *= (1 - 2 * np.mod(index, 2))  # If j odd, make m negative.

    n[index < 1] = -1  # Mark all indexes less than 1 as invalid

    return n, m


def orders2index(n: array_like[NP_INT_TYPE], m: array_like[NP_INT_TYPE] = 0) -> array_type[NP_INT_TYPE]:
    """
    Converts a Zernike order (radial degree n, azimuthal frequency m), to standard OSA/ANSI j-indices.

    When multiple values are specified, index will have the same shape as the inputs n and m.

    Invalid indices are marked as -1.

    See also the inverse operation: n, m = :py:func:``index2orders`(j)

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The standard Zernike index, j, or an ndarray thereof.
    """
    n = asarray(n, dtype=int)
    m = asarray(m, dtype=int)

    index = asarray((n * (n + 2) + m) / 2, dtype=int)

    index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return index


def orders2noll(n: array_like[NP_INT_TYPE], m: array_like[NP_INT_TYPE] = 0) -> array_type[NP_INT_TYPE]:
    """
    Converts a Zernike coordinate (radial degree n, azimuthal frequency m), to Noll indexes.

    When multiple values are specified, the Noll index will have the same shape as the inputs n and m.
    Invalid indices are marked as -1. The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF).
    J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.

    See also the inverse operation: n, m = :py:func:``noll2orders`(index)

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The Noll Zernike index, or an ndarray thereof.
    """
    n = np.array(n, dtype=int)
    m = np.array(m, dtype=int)

    index = asarray(n * (n + 1) / 2, dtype=int)  # number up to n-1
    index += np.abs(m) + (m == 0)  # correct number or one too low
    index += np.logical_and((m != 0), np.logical_xor(m < 0, np.mod(index, 2)))  # make index odd if m negative

    index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return index


def index2noll(index: array_like[NP_INT_TYPE]) -> array_type[NP_INT_TYPE]:
    """
    Converts a standard Zernike index or indices, js > 0, to Noll indexes.

    When multiple values are specified, m and n will have the same shape as the input js. Invalid indices are marked
    as -1. The standard OSA/ANSI ordering starts at 0. https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: js = :py:func:``noll2index``(js)

    :param index: The standard Zernike index, or an ndarray thereof.
    :return: The Noll Zernike index, or an ndarray thereof.
    """
    return orders2noll(*index2orders(index))


def noll2index(index: array_like[NP_INT_TYPE]) -> array_type[NP_INT_TYPE]:
    """
    Converts a Noll index or indices, js > 0, to a standard Zernike index or indices, js > 0.

    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Noll ordering starts counting at 1, not 0! The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF).
    J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.
    Invalid indices are marked as -1.

    See also the inverse operation: js = :py:func:``index2noll``(js)

    :param index: The Noll Zernike index, or an ndarray thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    return orders2index(*noll2orders(index))


def orders2fringe(n: array_like[NP_INT_TYPE], m: array_like[NP_INT_TYPE] = 0) -> array_type[NP_INT_TYPE]:
    """
    Converts a Zernike coordinate (radial degree n, azimuthal frequency m), to fringe (University of Arizona) indexes.

    When multiple values are specified, index will have the same shape as the inputs n and m.

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

    index = np.array((1 + (n + abs_m) // 2) ** 2 - 2 * abs_m + (m < 0))
    index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return index


def fringe2orders(index: array_like[NP_INT_TYPE]) -> tuple[array_type[NP_INT_TYPE], array_type[NP_INT_TYPE]]:
    """
    Converts a fringe (University of Arizona) index or indices, js > 0, to a degree tuple (radial m, azimuthal n).

    The degrees have the constraint 0 <= m <= n. When multiple values are specified, m and n will have the same shape
    as the input js. Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.

    See also the inverse operation: js = :py:func:``orders2fringe`(n, m)

    :param index: The Fringe Zernike index, or an ndarray thereof.
    :return: a tuple (n, m) of order subscripts or ndarrays thereof.
    """
    index = asarray(index, int)

    a = 2 * np.floor((index - 1) ** 0.5)
    s = (index - (a // 2) ** 2 - 1) // 2 * 2
    m_neg = (index - (a // 2) ** 2 - 1) % 2

    abs_m = (a - s) // 2
    m = asarray(abs_m * (1 - 2 * m_neg), int)
    n = asarray((a + s) // 2, int)

    n[index < 1] = -1  # Mark all indexes less than 1 as invalid

    return n, m


def index2fringe(index: array_like[NP_INT_TYPE]) -> array_type[NP_INT_TYPE]:
    """
    Converts a Zernike index or indices, js > 0, to fringe (University of Arizona) indexes.

    When multiple values are specified, m and n will have the same shape as the input js.
    Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.

    See also the inverse operation: js = :py:func:``fringe2index`(js)

    :param index: The standard Zernike index, or an ndarray thereof.
    :return: The Fringe Zernike index, or an ndarray thereof.
    """
    return orders2fringe(*index2orders(index))


def fringe2index(index: array_like[NP_INT_TYPE]) -> array_type[NP_INT_TYPE]:
    """
    Converts a fringe (University of Arizona) index or indices, js > 0, to a Zernike index or indices, js > 0.

    When multiple values are specified, m and n will have the same shape as the input js.
    Note that the Wyant indices start at 0, i.e. the Wyant index equals the Fringe index - 1.
    https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: j = :py:func:``index2fringe``(j)

    :param index: The Fringe Zernike index, or an ndarray thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    return orders2index(*fringe2orders(index))


class PolynomialBasis:
    """
    A class representing one of the Zernike basis polynomials, or an array thereof.

    Superpositions of weighted basis polynomials are represented by zernike.Polynomial.
    todo: refactor so that this inherits from Polynomial
    """

    def __init__(self,
                 index: array_like[NP_INT_TYPE] | None = None,
                 n: array_like[NP_INT_TYPE] | None = None,
                 m: array_like[NP_INT_TYPE] | None = None,
                 odd_and_even: bool = False,
                 ):
        """
        Constructs one of the Zernike basis polynomial or an array thereof.

        The Zernike basis polynomials form a sqrt(pi) * orthonormal basis on the unit disk, for 2x2-unit square,
        multiply with 4 / pi.
        The returned Zernike polynomials are themselves functions of polar coordinates (rho=0, phi=0)

        >>> PolynomialBasis(n=2, m=0)
        PolynomialBasis(4) = defocus
        >>> str(PolynomialBasis(n=2, m=0))
        'Z₂⁰'

        Returns the Zernike polynomial of radial order n and azimuthal frequency m, where m is between -n and n.

        >>> PolynomialBasis(4)
        PolynomialBasis(4) = defocus

        Returns the standard OSA/ANSI Zernike polynomial with standard coefficient j_index
        The first of which are:
        * piston,
        * tilt, tip,
        * oblique-astigmatism, defocus, vertical-astigmatism,
        * vertical-trefoil, vertical-coma, horizontal-coma,  horizontal-trefoil,
        * oblique-trefoil, oblique-quadrafoil, oblique-secondary-astigmatism,
        * spherical aberration, vertical-secondary-astigmatism vertical-quadrafoil, ...
        where the postscripts indicate the position of the extreme value on the pupil edge.

        When many polynomials need to computed, it will be more efficient to compute multiple polynomials in parallel.
        This function can handle rho and phi matrices and n and m vectors, while the option odd_and_even returns the
        even and odd polynomials as a complex result.

        .. code:: python

            result = PolynomialBasis(n=n, m=m, odd_and_even = True)

        For m >= 0, returns the even Zernike polynomial(cos) value as the real part, and the odd polynomial(sin)
        value as the imaginary part. For m < 0, the odd Zernike value is returned as the real part, and the even is
        returned as the imaginary part.

        See also: fit, Fit and Polynomial, index2orders(j), noll2orders(j), orders2index(n, m=0),
            and orders2noll(n, m=0)

        :param index: (optional) The standard (OSA/ANSI) index of the polynomial. This can a non-negative integer or
            an nd-array of such integers.
        :param n: (optional) The radial order of the polynomial. This can a non-negative integer or an nd-array of
            such integers.
        :param m: (optional) The azimuthal frequency of the polynomial. This can a integer <= n or an nd-array of
            such integers.
        :param odd_and_even: A boolean to indicate if the odd or even counterpart should also be returned. When True,
            the imaginary parts of the result contain the counterpart of the requested polynomial (default: False).
        """
        if index is not None:
            if n is None and m is None:
                n, m = index2orders(index)
            else:
                raise ValueError('When the j-index of the Zernike polynomial basis is specified, ' +
                                 'neither order n, nor order m, should be specified.',
                                 )
        elif n is None or m is None:
            raise ValueError('When the j-index of the Zernike polynomial basis is not specified, ' +
                             'both order n, and order m, should be specified.',
                             )

        self.__n = asarray(n, int)
        self.__m = asarray(m, int)

        self.odd_and_even = odd_and_even

    def __call__(self,
                 rho: array_like[NP_FLOAT_TYPE] = 0,
                 phi: array_like[NP_FLOAT_TYPE] = 0,
                 ) -> array_type[NP_FLOAT_TYPE]:
        """
        Returns the Zernike polynomial of order (n, m) evaluated in polar coordinates at rho and phi.

        The result is represented as a numpy ndarray of dimensions equal, or broadcastable, to the shape of rho
        (and theta), or higher dimensions when n and m are also vectors or arrays.
        The arrays: n, m, j, rho, and phi must be broadcastable.

        :param rho: The radian coordinate. When negative, phi is changed by pi.
            This can be a single number or an nd-array with shape that is broadcastable with the orders n and m of
            the polynomial.
        :param phi: The azimuthal coordinate [-pi, pi). This can be a single number or an nd-array with shape that is
            broadcastable with the orders n and m of the polynomial.
        :return: A numpy ndarray of dimensions equal to the shape of rho (and phi),
            or higher dimensions when n and m are also vectors or arrays.
        """
        return self.polar(rho=rho, phi=phi)

    @property
    def n(self) -> array_type[NP_INT_TYPE]:
        """Get the radial order of the Zernike polynomial."""
        return self.__n

    @n.setter
    def n(self, new_radial_order: array_like[NP_INT_TYPE]):
        """Set the radial order of the Zernike polynomial."""
        self.__n = asarray(new_radial_order, int)

    @property
    def m(self) -> array_type[NP_INT_TYPE]:
        """Get the azimuthal order of the Zernike polynomial."""
        return self.__m

    @m.setter
    def m(self, new_azimuthal_order: array_like[NP_INT_TYPE]):
        """Set the azimuthal order of the Zernike polynomial."""
        self.__m = asarray(new_azimuthal_order, int)

    @property
    def index(self) -> array_type[NP_INT_TYPE]:
        """Get the standard OSA/ANSI index of the Zernike polynomial."""
        return orders2index(n=self.n, m=self.m)

    @index.setter
    def index(self, new_index: array_like[NP_INT_TYPE]):
        """Set the standard OSA/ANSI index of the Zernike polynomial."""
        self.n, self.m = index2orders(new_index)

    @property
    def name(self) -> str:
        """The latin name of this basis Zernike polynomial."""
        def radial_multiplicity(_: int) -> str:
            """Returns a string that describes the radial multiplicity of a basis polynomial for a given number."""
            prefixes = '0-', 'prim', 'second', 'terti', 'quatern', 'quint', 'sext', 'sept', 'oct'
            return (prefixes[_] if _ < len(prefixes) else f'{_}-') + 'ary '

        def azimulthal_multiplicity(m: int) -> str:
            """
            Returns a string that describes the azimulthal multiplicity of a basis polynomial for a given number.
            """
            abs_m = abs(m)
            special_names = ['spherical', 'coma', 'astigmatism']
            if abs_m < len(special_names):
                return special_names[abs_m]
            prefixes = '0-', '1-', '2-', 'tre', 'quadra', 'penta', 'hexa', 'hepta', 'octa', 'nona', 'deca'
            return (prefixes[abs_m] if abs_m < len(prefixes) else f'{abs_m}-') + 'foil'

        def get_single_name(n: int, m: int) -> str:
            """Returns the description of single basis polynomial."""
            if orders2index(n, m) < 0:
                return 'undefined'
            if m == 0:
                if n == 0:
                    name = 'piston'
                elif n == 2:
                    name = 'defocus'
                else:  # Start counting from spherical
                    name = radial_multiplicity(n // 2 - 1) + azimulthal_multiplicity(m)
            elif abs(m) == 1:
                if n == 1:
                    name = 'tilt' if m < 0 else 'tip'
                else:
                    name = 'vertical ' if m < 0 else 'horizontal '
                    if n > 3:
                        name += radial_multiplicity((n - abs(m)) // 2)
                    name += azimulthal_multiplicity(m)
            else:
                if abs(m) % 2 == 0:
                    name = 'oblique ' if m < 0 else 'vertical '
                else:
                    name = 'vertical ' if m < 0 else ('horizontal ' if n > 3 else 'oblique ')
                if n > abs(m):
                    name += radial_multiplicity(1 + (n - abs(m)) // 2)
                name += azimulthal_multiplicity(m)
            return name

        def get_name_recursive(n: array_like[NP_INT_TYPE], m: array_like[NP_INT_TYPE]) -> str:
            """Returns a string with a (nested) list of PolynomialBasis-descriptions."""
            n_full, m_full = np.broadcast_arrays(n, m)
            if m_full.ndim > 0:
                return '[' + ', '.join(get_name_recursive(*slices) for slices in zip(n_full, m_full)) + ']'
            return get_single_name(n_full.item(), m_full.item())

        return get_name_recursive(self.n, self.m)

    @staticmethod  # TODO: may need caching
    def __polynomial_r_static(n: array_type[NP_INT_TYPE], m: array_type[NP_INT_TYPE],
                              rho: array_type[NP_FLOAT_TYPE],
                              ) -> array_type[NP_FLOAT_TYPE]:
        """
        Calculate the radial polynomial, for all rho in a matrix.

        Prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case
        the difference n - m is odd.

        :param n: A non-negative integer or array_like[NP_INT_TYPE] indicating the radial order.
        :param m: An integer or array_like[NP_INT_TYPE] indicating the azimuthal order.
        :param rho: An nd-array with the radial distances. Non-negativeness is enforced.

        :return: The polynomial values in an nd-array of the same shape as rho, but broadcasted with n and m.
        """
        n_m_dim: int = n.ndim

        # Expand the output to the shape of that of rho_i broadcasted with n and m
        if rho.ndim < 1:
            rho = rho[..., np.newaxis]
        output_shape = (*rho.shape[:rho.ndim - n_m_dim], *np.maximum(np.array(n.shape),
                                                                     np.array(rho.shape[rho.ndim - n_m_dim:]),
                                                                     ))
        calculation_shape = (*output_shape[:len(output_shape) - n_m_dim],
                             prod(output_shape[len(output_shape) - n_m_dim:]),
                             )
        rho = np.broadcast_to(rho, shape=output_shape)

        # Start with the first n_m_dim dimensions flattened
        result = np.zeros(shape=calculation_shape)
        rho = np.reshape(rho, shape=calculation_shape)
        for idx in range(n.size):
            n_i: int = int(n.ravel()[idx])
            m_i: int = int(m.ravel()[idx])
            rho_i = rho[..., idx]
            if (n_i - m_i) % 2 == 0:  # Skip odd differences, for these the result is zero.
                hd = (n_i - m_i) // 2
                hs = (n_i + m_i) // 2
                rho_pow = rho_i ** m_i
                rho_sqd = rho_i ** 2

                coefficients = (-1) ** hd * factorial_product_fraction(hs, (hd, m_i))
                result[..., idx] = coefficients * rho_pow
                for k in range(hd - 1, -1, -1):  # note the coefficients are from small powers to large
                    rho_pow *= rho_sqd  # For speedup: rho_pow = rho_i ** (n_i - 2 * coefficients)
                    coefficients *= - (n_i - k) * (k + 1) / (hs - k) / (hd - k)
                    result[..., idx] += coefficients * rho_pow

        return result.reshape(output_shape)

    def __polynomial_r(self, rho: array_type[NP_FLOAT_TYPE]):
        """
        Calculate the radial polynomial component, for all rho in a matrix.

        prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case the
        difference n - m is odd.

        :param rho: An nd-array with the radial distances. This array must have singleton. Non negativeness is enforced.

        :return: The polynomial values in an nd-array of the same shape as rho, but broadcasted with n and m.
        """
        return self.__polynomial_r_static(self.n, np.abs(self.m), np.abs(rho))

    def polar(self, rho: array_like[NP_FLOAT_TYPE] = 0, phi: array_like[NP_FLOAT_TYPE] = 0,
              ) -> array_type[NP_FLOAT_TYPE]:
        """
        Returns the Zernike polynomial of order (n, m) evaluated in polar coordinates at rho and phi.

        The result is represented as a numpy ndarray of dimensions equal, or broadcastable, to the shape of rho
        (and theta), or higher dimensions when n and m are also vectors or arrays.
        The arrays: n, m, j, rho, and phi must be broadcastable.

        When the basis consists of multiple polynomials, the right-most dimension of the input arguments is used to
        select the different polynomials. If that is not desired, add a dimension on the right.
        E.g. ``rho[..., np.newaxis]``

        :param rho: The radian coordinate. When negative, phi is changed by pi.
            This can be a single number or an nd-array with shape that is broadcastable with the orders n and m of
            the polynomial.
        :param phi: The azimuthal coordinate [-pi, pi). This can be a single number or an nd-array with shape that is
            broadcastable with the orders n and m of the polynomial.
        :return: A numpy ndarray of dimensions equal to the shape of rho (and phi),
            or higher dimensions when n and m are also vectors or arrays.
        """
        rho = asarray(rho, float)
        phi = asarray(phi, float)
        # Make orthogonal basis on unit disk (for 2x2 square, set everything outside unit disk to zero and multiply
        # by 4/pi). The norm of each basis vector is sqrt(pi), so that piston(rho, phi) = 1 everywhere.
        normalization = np.sqrt(2 * (self.n + 1) / (1 + (self.m == 0)))
        # Set the real part as requested, the imaginary part will be the odd-counterpart polynomial
        zernike_phase = self.m * (phi + np.pi * (rho < 0)) + (self.m < 0) * np.pi / 2
        zernike_phasor = np.exp(1j * zernike_phase) if self.odd_and_even else np.cos(zernike_phase)
        return normalization * self.__polynomial_r(rho) * zernike_phasor

    def cartesian(self, y: array_like[NP_FLOAT_TYPE], x: array_like[NP_FLOAT_TYPE]) -> array_type[NP_FLOAT_TYPE]:
        """
        The Zernike polynomial values as a function of the Cartesian axis.

        Coordinates are broadcast.
        :param y: The first coordinate (number or anything that can be converted to an array).
        :param x: The second coordinate (number or anything that can be converted to an array).
        :return: The Zernike polynomial values as an array of a shape equal to the broadcasted shape of `y` and `x`.
        """
        return self.polar(rho=np.sqrt(np.square(y) + np.square(x)), phi=np.arctan2(y, x))

    def polar_gradient(self,
                       rho: array_like[NP_FLOAT_TYPE] = 0,
                       phi: array_like[NP_FLOAT_TYPE] = 0,
                       ) -> array_type[NP_FLOAT_TYPE]:
        """
        Returns the gradient from polar coordinates.

        The first (left-most) dimension has size 2 with the partial derivatives in the order [d_rho, d_phi].
        """
        rho = asarray(rho, float)
        phi = asarray(phi, float)
        # Make orthogonal basis on unit disk (for 2x2 square, set everything outside unit disk to zero and multiply
        # by 4/pi). The norm of each basis vector is sqrt(pi), so that piston(rho, phi) = 1 everywhere.
        normalization = np.sqrt(2 * (self.n + 1) / (1 + (self.m == 0)))
        # Set the real part as requested, the imaginary part will be the odd-counterpart polynomial
        zernike_phase = self.m * (phi + np.pi * (rho < 0)) + (self.m < 0) * np.pi / 2
        if self.odd_and_even:
            zernike_phasor = np.exp(1j * zernike_phase)
            d_zernike_phasor = zernike_phasor * 1j * zernike_phase * self.m
        else:
            zernike_phasor = np.cos(zernike_phase)
            d_zernike_phasor = - np.sin(zernike_phase) * zernike_phase * self.m

        rho2 = rho ** 2
        rho2m1 = rho2 - 1

        dzdrho = ((2 * self.n * self.m * rho2m1 + (self.n - self.m) * (self.m + self.n * (2 * rho2 - 1))) *
                  self.__polynomial_r(np.abs(rho)) -
                  (self.n + self.m) * (self.n - self.m) * self.__polynomial_r(np.abs(rho))
                  ) / (2 * self.n * rho * rho2m1) * normalization * zernike_phasor
        dzdphi = normalization * self.__polynomial_r(np.abs(rho)) * d_zernike_phasor

        return np.stack([dzdrho, dzdphi])

    def __mul__(self, other: array_like[NP_FLOAT_TYPE]) -> Polynomial:
        """Create a Zernike ``Polynomial`` by scaling this ``PolynomialBasis``."""
        return Polynomial(indices=self.index, coefficients=other)

    def __rmul__(self, other: FLOAT_TYPE) -> Polynomial:  # type: ignore  # mypy bug
        """Create a Zernike ``Polynomial`` by scaling this ``PolynomialBasis`` from the right."""
        return self * other

    def __str__(self) -> str:
        """Return a compact representation of this polynomial as a unicode string."""
        return f'Z{script.sub(self.n)}{script.sup(self.m)}'

    def __repr__(self) -> str:
        """Return a representation of this polynomial as a string."""
        return f'{self.__class__.__name__}({self.index}) = {self.name}'


# Some definitions for convenience. For more names, check the PolynomialBasis.name property.
piston = PolynomialBasis(n=0, m=0)

tip = PolynomialBasis(n=1, m=-1)
tilt = PolynomialBasis(n=1, m=1)

oblique_astigmatism = PolynomialBasis(n=2, m=-2)
defocus = PolynomialBasis(n=2, m=0)
vertical_astigmatism = PolynomialBasis(n=2, m=-2)

vertical_trefoil = PolynomialBasis(n=3, m=-3)
vertical_coma = PolynomialBasis(n=3, m=-1)
horizontal_coma = PolynomialBasis(n=3, m=1)
oblique_trefoil = PolynomialBasis(n=3, m=3)

primary_spherical = PolynomialBasis(n=4, m=0)


class Polynomial:
    """A class to represent linear combinations of Zernike polynomials, which permit basic arithmetic operations."""

    def __init__(self, coefficients: array_like[NP_FLOAT_TYPE], indices: array_like[NP_INT_TYPE] | None = None):
        """
        Construct an object that represents superpositions of basis-Zernike polynomials.

        :param coefficients: The coefficients of the polynomials.
        :param indices: The standard indices of the polynomials (Default: all starting from 0).
        """
        self.__coefficients: array_type[NP_FLOAT_TYPE] = np.atleast_1d(asarray(coefficients))

        # The polynomial function objects that correspond to each index as listed by the `indices` property.
        self.__polynomial_basis: PolynomialBasis = PolynomialBasis(
            index=np.atleast_1d(
                asarray(indices) if indices is not None
                else np.arange(self.__coefficients.size).reshape(self.__coefficients.shape)
            ),
        )

    @property
    def basis(self) -> PolynomialBasis:
        """The polynomial function objects that correspond to each index as listed by the `indices` property."""
        return self.__polynomial_basis

    @property
    def coefficients(self) -> array_type[NP_FLOAT_TYPE]:
        """The coefficients corresponding to each of index as listed by the `indices` property."""
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, new_coefficients: array_like[NP_FLOAT_TYPE]):
        self.__coefficients = np.atleast_1d(asarray(new_coefficients))
        if self.coefficients.shape != self.basis.index.shape:
            log.warning('The number of coefficients of this Polynomial changed!')
            self.indices = None  # type: ignore  # Choose default indices

    @property
    def indices(self) -> array_type[NP_INT_TYPE]:
        """The standard ISO/ANSI indices of the basis polynomials."""
        assert self.basis.index.ndim > 0, f'{self.basis} has a scalar index. At least 1D expected.'
        return self.basis.index

    @indices.setter
    def indices(self, new_indices: array_like[NP_INT_TYPE] | None):
        if new_indices is None:
            new_indices_arr: array_type[NP_INT_TYPE] = np.arange(len(self.coefficients))
        else:
            new_indices_arr = np.atleast_1d(asarray(new_indices))
        self.__polynomial_basis = PolynomialBasis(index=new_indices_arr)

    @property
    def n(self) -> array_type[NP_INT_TYPE]:
        """The radial order of the Zernike polynomial."""
        return self.basis.n

    @n.setter
    def n(self, new_radial_order: array_like[NP_INT_TYPE]):
        """Set the radial order of the Zernike polynomial."""
        self.basis.n = np.atleast_1d(asarray(new_radial_order))

    @property
    def m(self) -> array_type[NP_INT_TYPE]:
        """The azimuthal order of the Zernike polynomial."""
        return self.basis.m

    @m.setter
    def m(self, new_azimuthal_order: array_like[NP_INT_TYPE]):
        """Set the azimuthal order of the Zernike polynomial."""
        self.basis.m = np.atleast_1d(asarray(new_azimuthal_order))

    @property
    def order(self) -> int:
        """The highest order of this Zernike polyonomial combination."""
        return 1 + int(np.amax(self.indices))

    def polar(self, rho: array_like[NP_FLOAT_TYPE] = 0, phi: array_like[NP_FLOAT_TYPE] = 0,
              ) -> array_type[NP_FLOAT_TYPE]:
        """
        Evaluate this polynomial at polar coordinates.

        :param rho: The radial coordinate between 0 and 1. Negative values are interpreted as a phase change of pi.
        :param phi: The angular coordinate in radians.

        :return: The value of the polynomial at the specified coordinates.
        """
        # Add one axis to the left for broadcasting over the self.__polynomials representation
        mat = self.__polynomial_basis(asarray(rho)[..., np.newaxis], asarray(phi)[..., np.newaxis])
        return mat @ self.coefficients

    def cartesian(self, y: array_like[NP_FLOAT_TYPE] = 0, x: array_like[NP_FLOAT_TYPE] = 0,
                  ) -> array_type[NP_FLOAT_TYPE]:
        """
        Evaluate this polynomial at Cartesian coordinates.

        :param y: The vertical coordinate between -1 and 1.
        :param x: The horizontal coordinate between -1 and 1.

        :return: The value of the polynomial at the specified coordinates.
        """
        return self.polar(rho=np.sqrt(np.square(y) + np.square(x)), phi=np.arctan2(y, x))

    def complex(self, z: array_like[NP_FLOAT_TYPE] | array_like[NP_COMPLEX_TYPE] = 0) -> array_type[NP_FLOAT_TYPE]:
        """
        Evaluate this polynomial in the complex plane, where the real and imaginary parts are the Cartesian coordinates.

        :param z: Complex numbers, where the real part is the horizontal, `x`, coordinate in the pupil with radius 1,
            and their imaginary part is the normalized `y`-coordinate (vertical).

        :return: The value of the polynomial at the specified coordinates.
        """
        return self.cartesian(y=asarray(z, complex).imag, x=asarray(z, complex).real)

    def __call__(self,
                 rho: array_like[NP_FLOAT_TYPE] | array_like[NP_COMPLEX_TYPE] | None = None,
                 phi: array_like[NP_FLOAT_TYPE] | None = None,
                 y: array_like[NP_FLOAT_TYPE] | None = None,
                 x: array_like[NP_FLOAT_TYPE] | None = None,
                 ) -> array_type[NP_FLOAT_TYPE]:
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
        if rho is None:
            assert x is not None and y is not None, 'Either x and y, or rho must be specified.'
            return self.cartesian(y=y, x=x)
        if phi is None:
            return self.complex(z=rho)
        return self.polar(rho=asarray(rho).real, phi=phi)

    def __add__(self, other: FLOAT_TYPE | Polynomial) -> Polynomial:
        """
        Return a polynomial that represents the sum of this and another polynomial.

        If the right-hand side is a number, it is interpreted as piston.
        """
        if not isinstance(other, Polynomial):
            other = piston * other
        assert_type(other, Polynomial)
        # DICT_TYPE = Dict[array_like[NP_INT_TYPE], array_like[NP_FLOAT_TYPE] | array_like[NP_COMPLEX_TYPE]]
        new_coefficients: Dict[int, float] = defaultdict(float)
        for _, c in zip(self.indices, self.coefficients):
            new_coefficients[int(_)] = float(c.item())  # copy
        for _, d in zip(other.indices, other.coefficients):
            _ = int(_)
            new_coefficients[_] += float(d)  # add
        combined_indices = sorted(new_coefficients.keys())
        combined_coefficients = [new_coefficients[ci] for ci in combined_indices]
        return Polynomial(coefficients=combined_coefficients, indices=combined_indices)

    def __radd__(self, other: FLOAT_TYPE) -> Polynomial:  # type: ignore  # mypy bug?
        """Return a new ``Polynomial`` that adds the specified piston to this ``Polynomial``."""
        return self + other

    def __neg__(self, other: FLOAT_TYPE | Polynomial) -> Polynomial:
        """Return the negated Polynomial, with the sign of all coefficients changed."""
        return Polynomial(coefficients=-self.coefficients, indices=self.indices)

    def __sub__(self, other: FLOAT_TYPE | Polynomial) -> Polynomial:
        """
        Return a polynomial that represents that difference of this and another polynomial.

        If the right-hand side is a number, it is interpreted as piston.
        """
        return self + (-other)  # type: ignore  # mypy bug?

    def __rsub__(self, other: FLOAT_TYPE) -> Polynomial:  # type: ignore  # mypy bug?
        """
        Return a new ``Polynomial`` that is the difference of the specified amount of piston and this ``Polynomial``.
        """
        return (-self) + other  # type: ignore  # mypy bug?

    def __mul__(self, other: FLOAT_TYPE) -> Polynomial:
        """Return a new Polynomial that equals this one multiplied by a scalar factor."""
        return Polynomial(coefficients=self.coefficients * other)

    def __rmul__(self, other: FLOAT_TYPE) -> Polynomial:  # type: ignore  # mypy bug
        """Return a new Polynomial that equals this one multiplied by a scalar factor."""
        return self * other

    def __truediv__(self, other: FLOAT_TYPE) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return self * (1 / other)

    def __imul__(self, other: FLOAT_TYPE):
        """In-place multiply (`*=`) this Polynomial by a scalar constant."""
        self.coefficients *= other
        return self

    def __idiv__(self, other: FLOAT_TYPE):
        """In-place divide (`/=`) this Polynomial by a scalar constant."""
        self.coefficients /= other
        return self

    def __str__(self) -> str:
        """Returns a string for the display of this polynomial."""
        descriptions = list[str]()
        for _, c in zip(self.indices, self.coefficients):
            if c != 0:
                c_str = '' if c == 1 else str(c)
                if len(descriptions) > 0 and not c_str.startswith('-'):
                    descriptions.append('+')
                descriptions.append(c_str)
                n, m = index2orders(_)
                descriptions.append(f'Z{script.sub(n)}{script.sup(m)}')
        if len(descriptions) > 0:
            return ''.join(descriptions)
        return str(self.coefficients[0]) if self.coefficients.size > 0 else '0'

    def __repr__(self) -> str:
        """Returns a string that is a complete description of this polynomial."""
        return f'{self.__class__.__name__}(coefficients={self.coefficients}, indices={self.indices})'


class Fit(Polynomial):
    """A class to fits Zernike polynomials up to the specified order."""

    def __init__(self,
                 z: array_like[NP_FLOAT_TYPE],
                 y: array_like[NP_FLOAT_TYPE] | None = None,
                 x: array_like[NP_FLOAT_TYPE] | None = None,
                 rho: array_like[NP_FLOAT_TYPE] | None = None,
                 phi: array_like[NP_FLOAT_TYPE] | None = None,
                 weight: array_like[NP_FLOAT_TYPE] | None = None,
                 order: INT_TYPE = 15,
                 ):
        """
        Construct an object to fit Zernike polynomials up to the given order.

        The coordinate arguments ``phi``, ``phi``, ``x``, and ``y``, can have dimension 0, 1, or 2.
        The arguments ``z`` and ``weight``, can have any dimension. The two right-most dimensions must broadcast with
        the coordinate arguments. Further arguments on the left are computed in parallel.

        :param z: The real function to fit, as an nd-array where the right-most dimensions are indexed by
            either ``x`` and ``y`` or by ``rho`` and ``phi``.
        :param y: The second Cartesian coordinate, which must broadcast with ``z`` and can be up to 2-dimensional.
            Default: covering the range [-1, 1)
        :param x: The first Cartesian coordinate, which must broadcast with ``z`` and can be up to 2-dimensional.
            Default: covering the range [-1, 1)
        :param rho: Alternative radial coordinate when not using Cartesian coordinates, which must broadcast with ``z``
            and can be up to 2-dimensional.
        :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates, which must broadcast
            with ``z`` and can be up to 2-dimensional.
        :param weight: An nd-array with per-value weights for the fit, which must broadcast with ``z`` and can have any
            dimension. Default: None = uniform weighting within the unit disk (including its edge).
        :param order: The number of polynomial terms to consider.
        """
        super().__init__(coefficients=[])

        z_arr: array_type[NP_FLOAT_TYPE] = asarray(z, float)

        rho_arr: array_type[NP_FLOAT_TYPE]
        phi_arr: array_type[NP_FLOAT_TYPE]
        if rho is None and phi is None:  # cartesian, convert to radial
            if x is None and y is None:
                z_arr = np.atleast_2d(z_arr)
                y, x = np.linspace(-1, 1, z_arr.shape[-2])[:, np.newaxis], np.linspace(-1, 1, z_arr.shape[-1])

            rho_arr, phi_arr = cart2pol(y, x)  # type: ignore
        else:
            rho_arr = asarray(rho)
            phi_arr = asarray(phi)

        coordinate_shape: tuple[int, ...] = np.broadcast_shapes(rho_arr.shape, phi_arr.shape)
        if weight is None:
            # Polar     => proportional to rho in unit disk, 0 outside
            # Cartesian => uniform in unit disk, 0 outside
            inside = np.broadcast_to(rho_arr <= 1, shape=coordinate_shape)
            weight = inside.astype(float)

        self.__rho: array_type[NP_FLOAT_TYPE] = rho_arr
        self.__phi: array_type[NP_FLOAT_TYPE] = phi_arr
        self.__z: array_type[NP_FLOAT_TYPE] = z_arr
        self.__weight: array_type[NP_FLOAT_TYPE] = asarray(weight)

        # Lazily calculated
        self.__error: array_type[NP_FLOAT_TYPE]
        self.order = int(order)  # Calculates the above as well

    @property
    def order(self) -> int:
        """The total number of basis Zernike polynomials for the fit."""
        return self.coefficients.size

    @order.setter
    def order(self, new_order: INT_TYPE):
        """
        Sets the order and fits Zernike basis polynomials up to it.

        The fitting is a weighted least-squared fitting at the specified coordinates.

        :param new_order: The number of polynomials to fit.
        """
        self.coefficients = np.zeros(new_order)  # Also determine the polynomials in the super class

        calc_shape = np.broadcast_shapes(self.__rho.shape, self.__phi.shape, self.__z.shape)

        # The right-most dimension is for the basis index
        basis_vectors = self.basis(self.__rho[..., np.newaxis], self.__phi[..., np.newaxis])
        basis_vectors *= self.__weight[..., np.newaxis]
        # make it a stack of matrices for np.linalg.lstsq-loop
        basis_vectors = basis_vectors.reshape((-1, prod(basis_vectors.shape[-3:-1]), self.order))
        zs: array_type[NP_FLOAT_TYPE] = self.__z.reshape((-1, prod(self.__z.shape[-2:]), 1))

        coefficient_list: list[array_type[NP_FLOAT_TYPE]] = list[array_type[NP_FLOAT_TYPE]]()
        error_list: list[float] = list[float]()
        for bvs, z in zip(basis_vectors, zs):
            coefficients, sqd_residuals, rank, s = np.linalg.lstsq(bvs, z, rcond=None)
            coefficient_list.append(coefficients)
            error_list.append(np.linalg.norm(sqd_residuals ** 0.5) / np.sqrt(np.sum(self.__weight.ravel())))

        # Set the coefficients of the super() Polynomial
        self.coefficients = np.stack(coefficient_list).reshape((*calc_shape[:-2], self.order))

        # Store the l2-fitting error for future reference
        self.__error = asarray(error_list).reshape(self.coefficients.shape[:-2])

    @property
    def contravariant(self) -> array_type[NP_FLOAT_TYPE]:
        """
        The contravariant coefficients of the Zernike polynomial fit.

        These are the coefficients that multiply the basis Zernike polynomials that make up the fitted polynomial.
        These are computed by (re)setting the order property.
        """
        return self.coefficients

    @property
    def covariant(self) -> array_type[NP_FLOAT_TYPE]:
        """
        The covariant coefficients of this fit.

        These are the projections of the specified function onto the (weighted) Zernike basis polynomials.
        Without weights and continuous uniform sampling, these are the same as the contravariant coordinates.
        """
        coefficients = np.zeros(shape=(*self.__z.shape[:-2], self.order))
        nb_samples = prod(np.broadcast_shapes(self.__rho.shape, self.__phi.shape))
        for idx in range(self.order):
            basis_vector = (PolynomialBasis(idx)(self.__rho, self.__phi) * self.__weight)
            basis_vector = basis_vector.reshape((*basis_vector.shape[:-3], nb_samples))
            coefficients[..., idx] = basis_vector[np.newaxis, :] @ self.__z.reshape(
                (*self.__z.shape[:-2], nb_samples, 1),
            )
        return coefficients

    @property
    def error(self) -> array_type[NP_FLOAT_TYPE]:
        """
        The root-mean-square (RMS) fitting error between `f` and `z`.

        ||(z - f) w|| / sqrt(n), where `w` are the optional weights and `n` is the number of sample points.
        This is computed by (re)setting the order property.
        """
        return self.__error

    def __str__(self) -> str:
        """The representation of this object as a string."""
        return f'{self.__class__.__name__}({self.coefficients})'


def fit(z: array_like[NP_FLOAT_TYPE],
        y: array_like[NP_FLOAT_TYPE] | None = None,
        x: array_like[NP_FLOAT_TYPE] | None = None,
        rho: array_like[NP_FLOAT_TYPE] | None = None,
        phi: array_like[NP_FLOAT_TYPE] | None = None,
        weight: array_like[NP_FLOAT_TYPE] | None = None,
        order: INT_TYPE = 15,
        ) -> Fit:
    """
    Fits Zernike polynomial up to the given order and returns a Fit object.

    TODO: Remove this function as it does not seem to give any benefit over using the Fit class directly.

    The fit object holds the coefficients, the polynomial, and the fitting error.

    See also the :py:class:``Fit`` class.

    :param z: The real function to fit, as an nd-array where the right-most dimensions are indexed by
            either ``x`` and ``y`` or by ``rho`` and ``phi``.
    :param y: The second Cartesian coordinate, which must broadcast with ``z`` and can be up to 2-dimensional.
        Default: covering the range [-1, 1)
    :param x: The first Cartesian coordinate, which must broadcast with ``z`` and can be up to 2-dimensional.
        Default: covering the range [-1, 1)
    :param rho: Alternative radial coordinate when not using Cartesian coordinates, which must broadcast with
        ``z`` and can be up to 2-dimensional.
    :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates, which must broadcast with ``z``
        and can be up to 2-dimensional.
    :param weight: An nd-array with per-value coefficients for the fit, which must broadcast with ``z``.
        Default: None = uniform weighting on the unit disk, weighted by rho in case of polar coordinate specification.
    :param order: The number of polynomial terms to consider.

    :return: The Fit object representing the polynomial.
    """
    return Fit(z=z, y=y, x=x, rho=rho, phi=phi, weight=weight, order=order)
