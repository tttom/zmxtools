from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Sequence, Generator

import numpy as np

from zmxtools.utils import script
from zmxtools.utils.array import NP_FLOAT_TYPE, array_like, array_type, asarray
from zmxtools.utils.polar import cart2pol

__all__ = ['Polynomial']


class Polynomial:
    """
    A class to represent Cartesian multivariate polynomials.

    Generic exponents are allowed, both positive and negative, i.e. as a Laurent polynomial.
    https://en.wikipedia.org/wiki/Laurent_polynomial
    """

    def __init__(self, coefficients: array_like[NP_FLOAT_TYPE], labels: Sequence[str] = (),
                 exponents: Sequence[Sequence[int | float | complex]] = (),
                 ):
        """
        Construct a multivariate polynomial object that can be evaluated at specific points or array's thereof.

        :param coefficients: The coefficients as a multidimensional array with the N-th dimension corresponding to the
            N-th independent coordinate.
        :param labels: The names or symbols of the independent variables in order. This is used to display or to select
            the arguments by name. By default, x₀, x₁, x₂, x₃, ... is used.
        :param exponents: The optional exponents of the polynomial. By default, these are just 0, 1, 2, ...
        """
        self.__coefficients: array_type[NP_FLOAT_TYPE] = asarray(coefficients, float)
        self.coefficients = self.coefficients
        self.__symbols: tuple[str, ...] = ()
        self.labels = labels
        self.__exponents = tuple[Sequence[int | float | complex]]()
        self.exponents = exponents

    def __call__(self, *args: array_like[NP_FLOAT_TYPE], **kwargs: array_like[NP_FLOAT_TYPE],
                 ) -> array_type[NP_FLOAT_TYPE]:
        """
        The evaluated value of this polynomial at the specified coordinates.

        These can be specified in order of the labels or by label name. The coordinates are broadcast as necessary.
        Keyword arguments override non-named arguments.

        :param args: The coordinates in order of the symbols.
        :param kwargs: (optional) The coordinates by their label.

        :return: The polynomial value for each argument coordinate. The result has a shape that is equal to the
            broadcasted dimensions of the arguments.
        """
        for s in kwargs:
            assert s in self.labels, f'Unknown coordinate symbol, {s}. Must be one of {self.labels}.'

        # Convert arguments to standard form. Default to 0
        arg_dict: Dict[str, array_type[NP_FLOAT_TYPE]] = defaultdict[str, array_type[NP_FLOAT_TYPE]](lambda: asarray(0))
        for symbol, arg in zip(self.labels, args):
            arg_dict[symbol] = asarray(arg)
        for symbol2, arg2 in kwargs.items():
            arg_dict[symbol2] = asarray(arg2)

        arguments = [arg_dict[label] for label in self.labels]
        while len(arguments) < self.ndim:  # Assume that missing arguments are 0.
            arguments.append(asarray(0))

        calculation_axes = tuple(range(-self.coefficients.ndim, 0))  # The axes of the multi-variate polynomial

        def calc_product_rec(coordinates: Sequence[array_type[NP_FLOAT_TYPE]],
                             exponents: Sequence[Sequence[int | float | complex]],
                             ) -> array_type[NP_FLOAT_TYPE]:
            coordinate = np.expand_dims(coordinates[0], axis=calculation_axes)
            exponents_for_this_axis = np.expand_dims(exponents[0], axis=tuple(range(-(len(exponents) - 1), 0)))
            result = coordinate ** exponents_for_this_axis
            if len(exponents) > 1:
                result = result * calc_product_rec(coordinates[1:], exponents[1:])
            return result

        # TODO: Memory could likely be saved here.

        return np.sum(self.coefficients * calc_product_rec(arguments, self.exponents), axis=calculation_axes)

    @property
    def coefficients(self) -> array_type[NP_FLOAT_TYPE]:
        """
        The multi-variate polynomial's coefficients as a multi-dimensional array.

        The array's dimensions are in the same order as the symbols. This array has a dimension equal to the number
        of variables and a shape, equal to the number of exponents for each variable.
        """
        return self.__coefficients.copy()

    @coefficients.setter
    def coefficients(self, new_coefficients: array_like[NP_FLOAT_TYPE]):
        self.__coefficients = asarray(new_coefficients)

    @property
    def labels(self) -> Sequence[str]:
        """The symbols that are used to represent this as a str. Their number must equal self.coefficients.ndim."""
        return self.__symbols

    @labels.setter
    def labels(self, new_symbols: Sequence[str]):
        symbols = list(new_symbols)
        assert len(symbols) == len(set(symbols)), f'No duplicate symbols are allowed. Got {symbols}.'
        for _ in range(len(symbols), self.ndim):
            symbols.append('x' + script.sub(_))
        self.__symbols = tuple(symbols)

    @property
    def exponents(self) -> Sequence[Sequence[int | float | complex]]:
        """The exponent for each coefficient. Its shape must equal self.shape."""
        return self.__exponents

    @exponents.setter
    def exponents(self, new_exponents: Sequence[Sequence[int | float | complex]]):
        exponents = list(new_exponents)
        for variable_index, exponent in enumerate(exponents):  # Make sure that sufficient exponents are specified
            if len(exponents[variable_index]) < self.shape[variable_index]:
                exponents[variable_index] = (
                    *exponent,
                    *range(len(exponent), self.shape[variable_index]),
                )
            elif len(exponents[variable_index]) > self.shape[variable_index]:
                raise ValueError(f'The number of exponents, {len(exponent)},' +
                                 f' for {self.labels[variable_index]} should match the number of coefficients, ' +
                                 f'{self.shape[variable_index]}.',
                                 )
            else:
                exponents[variable_index] = tuple(exponent)
        for var_index in range(len(exponents), self.ndim):  # Add default exponents for the remaining dimensions
            exponents.append(tuple(range(self.shape[var_index])))
        self.__exponents = tuple(exponents)

    @property
    def ndim(self) -> int:
        """The number of independent variables of this ``Polynomial``."""
        return self.coefficients.ndim

    @property
    def shape(self) -> Sequence[int]:
        """
        The number of exponents considered for each independent variable.

        Using the default exponents, these are the highest orders of the ``Polynomial``.
        """
        return self.coefficients.shape

    def grad(self) -> Sequence[Polynomial]:
        """
        Calculate the gradient of this ``Polynomial``.

        The partial derivatives are listed in the order of ``self.labels``.
        """
        result: List[Polynomial] = list[Polynomial]()
        for axis, exponents in enumerate(self.exponents):
            non_zero_exponents = [_ != 0 for _ in exponents]
            coefficients = self.coefficients.swapaxes(axis, -1)
            coefficients = coefficients[..., non_zero_exponents]  # Drop the vanishing exponents of axis _
            exponents = [e for e, nz in zip(exponents, non_zero_exponents) if nz]
            derivative_coefficients = (coefficients * exponents).swapaxes(-1, axis)
            derivative_exponents = list(self.exponents)
            derivative_exponents[axis] = [_ - 1 for _ in exponents]
            result.append(Polynomial(coefficients=derivative_coefficients,
                                     labels=self.labels,
                                     exponents=derivative_exponents,
                                     ))
        return result

    def __add__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Returns the sum of two polynomials, or this ``Polynomial`` and a constant value."""
        if not isinstance(other, Polynomial):
            other = self.__class__(other)  # A scalar

        coefficients: array_type[NP_FLOAT_TYPE] = self.coefficients
        symbols = list(self.labels)
        exponents = list(self.exponents)
        other_coefficients = other.coefficients
        other_symbols = list(other.labels)
        other_exponents = list(other.exponents)

        # Extend symbols and ndims of coefficients to include other_symbols and other_coefficients
        for other_symbol, other_exponent in zip(other_symbols, other_exponents):
            if other_symbol not in symbols:
                coefficients = coefficients[..., np.newaxis]
                symbols.append(other_symbol)
                exponents.append(other_exponent)
        # Extend other_coefficients to match
        for symbol, exponent in reversed(tuple(zip(symbols, exponents))):
            if symbol not in other_symbols:
                other_coefficients = other_coefficients[np.newaxis]
                other_symbols.insert(0, symbol)
                other_exponents.insert(0, exponent)
        other_coefficients = other_coefficients.transpose([other_symbols.index(_) for _ in symbols])
        other_exponents = [other_exponents[other_symbols.index(_)] for _ in symbols]
        del other_symbols

        # Extend the shape of coefficients and pad the coefficients
        def pad(arr: array_type[NP_FLOAT_TYPE], nb_new: int, axis: int) -> array_type[NP_FLOAT_TYPE]:
            pad_shape = list(arr.shape)
            pad_shape[axis] = nb_new
            return np.concatenate([arr, np.zeros_like(arr, shape=pad_shape)], axis=axis)
        for _ in range(coefficients.ndim):
            # Pad higher orders
            # TODO: Check overlap of exponents!
            extra_orders = other_coefficients.shape[_] - coefficients.shape[_]
            if extra_orders > 0:
                coefficients = pad(coefficients, extra_orders, axis=_)
                exponents[_] = [*exponents[_], *other_exponents[_][len(exponents[_]):]]
            elif extra_orders < 0:
                other_coefficients = pad(other_coefficients, -extra_orders, axis=_)
                other_exponents[_] = [*other_exponents[_], *exponents[_][len(other_exponents[_]):]]  # TODO: needed?

        coefficients += other_coefficients

        return self.__class__(coefficients=coefficients, labels=symbols, exponents=exponents)

    def __radd__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Add this ``Polynomial`` to another polyonmial or a number on the left or the right."""
        if not isinstance(other, Polynomial):
            other = self.__class__(other)
        return other + self

    def __neg__(self) -> Polynomial:
        """Negate the values of this ``Polynomial``."""
        return self.__class__(-self.coefficients, labels=self.labels, exponents=self.exponents)

    def __sub__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Return the difference of this ``Polynomial`` and another."""
        return self + (-other)

    def __rsub__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Subtract this ``Polynomial`` from the number on its right."""
        if not isinstance(other, Polynomial):
            other = self.__class__(other)
        return other - self

    def __mul__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Scale this ``Polynomial`` by a scalar or multiply it with another Polynomial."""
        if not isinstance(other, Polynomial):
            return self.__class__(coefficients=self.coefficients * other, labels=self.labels, exponents=self.exponents)
        raise NotImplementedError

    def __rmul__(self, right: int | float | complex) -> Polynomial:
        """Scale this ``Polynomial`` with the number on its right."""
        return Polynomial(coefficients=right * self.coefficients, labels=self.labels, exponents=self.exponents)

    def __truediv__(self, left: int | float | complex) -> Polynomial:
        """Divide this ``Polynomial`` using the `/` operation."""
        return self * (1 / left)

    def __str__(self) -> str:
        """Format this polynomial as a unicode string."""
        def format_factor(symbol: str, exponent: int | float | complex) -> str:
            result = '' if exponent == 0 else str(symbol)
            if exponent not in {0, 1}:
                result += script.sup(exponent)
            return result

        def format_coefficient(coefficient: int | float | complex, product_str: str) -> str:
            """
            Format the coeffcient number in front of a polynomial term.

            :param coefficient: The numerical value to format.
            :param product_str: The str description of the product element that is scaled by the coefficient.

            :return: A string representation of the coefficient.
            """
            if coefficient.real == 0:
                match int(coefficient.imag):
                    case -1:
                        return ' - i'
                    case 0:
                        return ' + 0.0'
                    case 1:
                        return ' + i'
                return f'{coefficient.imag:+}i'.replace('+', ' + ').replace('-', ' - ')
            elif coefficient.imag == 0:  # but coefficient.real != 0
                if len(product_str) == 0 or abs(coefficient) != 1:
                    return f'{coefficient.real:+}'.replace('+', ' + ').replace('-', ' - ')
                elif coefficient == -1:
                    return ' - '
                return ' + '
            # both coefficient.real != 0 and coefficient.imag != 0:
            return f' + ({coefficient.real}{coefficient.imag:+})'

        products = itertools.product(*([format_factor(s, e) for e in self.exponents[_]]
                                       for _, s in enumerate(self.labels)
                                       ),
                                     )
        products_strs: Generator[str, None, None] = (''.join(_) for _ in products)
        terms = [format_coefficient(c.item(), p) + p
                 for c, p in zip(self.coefficients.ravel(), products_strs) if c != 0
                 ]
        if len(terms) == 0:
            return '0.0'
        return (''.join(terms)).strip(' +')

    def __repr__(self) -> str:
        """The representation of this polynomial as a string."""
        return f'{self.__class__.__name__}({self.coefficients}, {self.labels}, {self.exponents})'

    def __hash__(self) -> int:
        """Hash value for dictionaries. Functionally identical polynomials return the same number."""
        return hash(repr(self))

    def __eq__(self, other: Polynomial | object) -> bool:
        """Returns True if these polynomials are functionally identical."""
        return bool(
            isinstance(other, Polynomial) and
            self.shape == other.shape and np.all(self.coefficients == other.coefficients) and
            all(s == o for s, o in zip(self.labels, other.labels)) and
            all(tuple(s) == tuple(o) for s, o in zip(self.exponents, other.exponents))
        )


class PolarPolynomial(Polynomial):
    """A class to represent polynomials in polar coordinates."""

    def __init__(self, coefficients: array_like[NP_FLOAT_TYPE],
                 labels: Sequence[str] = ('ρ', 'ϕ'),
                 exponents: Sequence[Sequence[int | float | complex]] = (),
                 ):
        """
        A polynomial of the form Σₙᵐ aₙᵐ ρⁿ cos(mϕ) + Σₙᵐ bₙᵐ ρⁿ sin(mϕ) =  Re Σₙᵐ cₙᵐ ρⁿ exp(-imϕ).

        The coefficients cₙᵐ = aₙᵐ + ibₙᵐ, can be complex.

        :param coefficients: The coefficients as a multi-dimensional array with the N-th dimension corresponding to the
            N-th independent coordinate.
        :param labels: The names or symbols of the independent variables in order. This is used to display or to select
            the arguments by name. By default, the labels x₀, x₁, x₂, x₃, ... is used.
        :param exponents: The optional exponents of the polynomial. By default, these are 0, 1, 2, ...
        """
        cartesian_labels = (labels[0], f'exp(-i{labels[1]})')  # or 'e⁻ⁱᵠ'
        super().__init__(coefficients=coefficients, labels=cartesian_labels, exponents=exponents)

    def __call__(self, rho: array_like[NP_FLOAT_TYPE] = 0, phi: array_like[NP_FLOAT_TYPE] = 0,
                 *args: array_like[NP_FLOAT_TYPE], **kwargs: array_like[NP_FLOAT_TYPE],
                 ) -> array_type[NP_FLOAT_TYPE]:
        """
        Calculates the ``PolarPolynomial``'s values at the specified polar coordinates.

        The coordinates are broadcast as necessary.

        :param rho: The radial coordinates.
        :param phi: The azimuthal coordinates in radians.
        :param args: for typing only. Ignored here.
        :param kwargs: for typing only. Ignored here.

        :return: The values at the specified coordinates.
        """
        return super().__call__(rho, np.exp(-1j * asarray(phi)))

    def cartesian(self, y: array_like[NP_FLOAT_TYPE], x: array_like[NP_FLOAT_TYPE]) -> array_type[NP_FLOAT_TYPE]:
        """Compute the values at Cartesian coordinates. The coordinates are broadcast as necessary."""
        rho, phi = cart2pol(y, x)
        return self(rho, phi)

    def cartesian_grad(self, y: array_like[NP_FLOAT_TYPE], x: array_like[NP_FLOAT_TYPE], axis: int = 0,
                       ) -> array_type[NP_FLOAT_TYPE]:
        g = self.grad()

        rho, phi = cart2pol(y, x)
        epsilon = 1e-6
        at_origin = rho < epsilon  # At the origin phi should be 0

        phasors = np.exp(-1j * phi)
        df_drho, df_dphi = g[0](rho, phasors).real, (-1j * phasors * g[1](rho, phasors)).real

        df_dphirho = ((1 - at_origin) * df_dphi / (rho + at_origin) +
                      at_origin * (-1j * phasors * g[1].grad()[0](0, phasors)).real
                      )  # Apply l'Hopital if necessary
        c, s = np.cos(phi), np.sin(phi)
        df_dx = c * df_drho - s * df_dphirho
        df_dy = s * df_drho + c * df_dphirho

        return np.stack([df_dy, df_dx], axis=axis)
