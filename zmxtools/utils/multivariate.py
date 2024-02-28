from __future__ import annotations

from collections import defaultdict
import itertools
import numpy as np
from typing import Optional, Callable, Sequence, Tuple, Dict, List

from zmxtools.utils.array import array_like, asarray, array_type
from zmxtools.utils import script


__all__ = ["Polynomial"]


class Polynomial(Callable):
    """A class to represent Cartesian multivariate polynomials."""
    def __init__(self,
                 coefficients: array_like,
                 symbols: Sequence[str] = tuple[str](),
                 exponents: Sequence[Sequence[int | float | complex]] = tuple[Sequence[int | float | complex]]()):
        self.__coefficients = None
        self.coefficients = coefficients
        self.__symbols = tuple[str]()
        self.symbols = symbols
        self.__exponents = tuple[Sequence[int | float | complex]]()
        self.exponents = exponents

    @property
    def coefficients(self) -> array_type:
        """
        The multi-variate polynomial's coefficients as a multi-dimensional array with dimensions in the same order
        as the symbols. This array has a dimension equal to the number of variables and a shape, equal to the number of
        exponents for each variable.
        """
        return self.__coefficients.copy()

    @coefficients.setter
    def coefficients(self, new_coefficients: array_like):
        self.__coefficients = asarray(new_coefficients)

    @property
    def symbols(self) -> Sequence[str]:
        """
        The symbols that are used to represent this as a str. Their number must equal self.coefficients.ndim
        """
        return self.__symbols

    @symbols.setter
    def symbols(self, new_symbols: Sequence[str]):
        symbols = list(new_symbols)
        assert len(symbols) == len(set(symbols)), f"No duplicate symbols are allowed. Got {symbols}."
        for _ in range(len(symbols), self.ndim):
            symbols.append('x'+script.sub(_))
        self.__symbols = tuple(symbols)

    @property
    def exponents(self) -> Sequence[Sequence[int | float | complex]]:
        """
        The exponent for each coefficient. Its shape must equal self.shape.
        """
        return self.__exponents

    @exponents.setter
    def exponents(self, new_exponents: Sequence[Sequence[int | float | complex]]):
        exponents = list(new_exponents)
        for variable_index in range(len(exponents)):  # Make sure that sufficient exponents are specified
            if len(exponents[variable_index]) < self.shape[variable_index]:
                exponents[variable_index] = tuple(
                    *exponents[variable_index],
                    *range(len(exponents[variable_index]), self.shape[variable_index])
                )
            elif len(exponents[variable_index]) > self.shape[variable_index]:
                raise ValueError(f"The number of exponents, {len(exponents[variable_index])}, for {self.symbols[variable_index]} should match the number of coefficients, {self.shape[variable_index]}.")
            else:
                exponents[variable_index] = tuple(exponents[variable_index])
        for variable_index in range(len(exponents), self.ndim):  # Add default exponents for the remaining dimensions
            exponents.append(tuple(range(self.shape[variable_index])))
        self.__exponents = tuple(exponents)

    @property
    def ndim(self) -> int:
        return self.coefficients.ndim

    @property
    def shape(self) -> Sequence[int]:
        return self.coefficients.shape

    def __call__(self, *args: array_like, **kwargs: array_like) -> array_type:
        """
        The evaluated value of this polynomial. The result has a shape that is equal to the broadcasted dimensions of
        the arguments. Keyword arguments override non-named arguments.

        :param args: The coordinates in order of the symbols.
        :param kwargs: (optional) The named coordinates.

        :return: The polynomial value for each argument coordinate.
        """
        for s in kwargs:
            assert s in self.symbols, f"Unknown coordinate symbol, {s}. Must be one of {self.symbols}."

        # Convert arguments to standard form
        arg_dict: Dict[str, array_type] = dict[str, array_type]()
        for symbol, arg in zip(self.symbols, args):
            arg_dict[symbol] = asarray(arg)
        for symbol, arg in kwargs.items():
            arg_dict[symbol] = asarray(arg)

        arguments = [arg_dict[symbol] for symbol in self.symbols]
        # assert len(arguments) == len(self.symbols), f"Expected exactly one argument for each of {self.symbols}, got {len(arguments)}."
        while len(arguments) < self.ndim:  # Assume that missing arguments are 0.
            arguments.append(asarray(0.0))

        calculation_axes = tuple(range(-self.coefficients.ndim, 0))  # The axes of the multi-variate polynomial

        def calc_product_rec(coordinates, exponents):
            coordinate = np.expand_dims(coordinates[0], axis=calculation_axes)
            exponents_for_this_axis = np.expand_dims(exponents[0], axis=tuple(range(-(len(exponents) - 1), 0)))
            result = coordinate ** exponents_for_this_axis
            if len(exponents) > 1:
                result = result * calc_product_rec(coordinates[1:], exponents[1:])
            return result

        return np.sum(self.coefficients * calc_product_rec(arguments, self.exponents), axis=calculation_axes)

    @property
    def gradient(self) -> Sequence[Polynomial]:
        result: List[Polynomial] = list[Polynomial]()
        for axis, exponents in enumerate(self.exponents):
            non_zero_exponents = [_ != 0 for _ in exponents]
            coefficients = self.coefficients.swapaxes(axis, -1)
            coefficients = coefficients[..., non_zero_exponents]  # Drop the vanishing exponents of axis _
            exponents = np.asarray(exponents)[non_zero_exponents]
            derivative_coefficients = (coefficients * exponents).swapaxes(-1, axis)
            derivative_exponents = list(self.exponents)
            derivative_exponents[axis] = exponents - 1
            result.append(Polynomial(coefficients=derivative_coefficients,
                                     symbols=self.symbols,
                                     exponents=derivative_exponents))
        return result

    def __add__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            other = Polynomial(other)  # A scalar

        coefficients = self.coefficients
        symbols = list(self.symbols)
        exponents = list(self.exponents)
        other_coefficients = other.coefficients
        other_symbols = list(other.symbols)
        other_exponents = list(other.exponents)

        # Extend symbols and ndims of coefficients to include other_symbols and other_coefficients
        new_axes = []
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
        def pad(arr: array_type, nb_new: int, axis: int) -> array_type:
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

        return Polynomial(coefficients=coefficients, symbols=symbols, exponents=exponents)

    def __radd__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            other = Polynomial(other)
        return other + self

    def __neg__(self) -> Polynomial:
        return Polynomial(-self.coefficients, symbols=self.symbols, exponents=self.exponents)

    def __sub__(self, other: Polynomial | int | float | complex) -> Polynomial:
        return self + (-other)

    def __rsub__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            other = Polynomial(other)
        return other - self

    def __mul__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            return Polynomial(coefficients=self.coefficients * other, symbols=self.symbols, exponents=self.exponents)
        else:
            raise NotImplementedError

    def __rmul__(self, other: int | float | complex) -> Polynomial:
        return Polynomial(coefficients=other * self.coefficients, symbols=self.symbols, exponents=self.exponents)

    def __truediv__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            return Polynomial(coefficients=self.coefficients / other, symbols=self.symbols, exponents=self.exponents)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        def format_factor(symbol: str, exponent: int | float | complex):
            result = str(symbol) if exponent != 0 else ""
            if exponent != 0 and exponent != 1:
                result += script.super(exponent)
            return result

        def format_coefficient(coefficient: complex, product_str: str):
            if coefficient.real != 0 and coefficient.imag != 0:
                return f"+({coefficient.real}{coefficient.imag:+})"
            elif coefficient.real != 0:  # coefficient.imag == 0
                if product_str == "" or abs(coefficient) != 1:
                    return f"{coefficient.real:+}"
                else:
                    return "+" if coefficient == 1 else "-"

            else:  # coefficient.imag != 0:
                return f"{coefficient.imag:+}i"

        products = itertools.product(*([format_factor(s, e) for e in self.exponents[_]] for _, s in enumerate(self.symbols)))
        products = ("".join(_) for _ in products)
        terms = [format_coefficient(c, p) + "".join(p) for c, p in zip(self.coefficients.ravel(), products)
                 if c != 0]
        if len(terms) > 0:
            if terms[0].startswith("+"):
                terms[0] = terms[0][1:]
            result = "".join(terms)
            result = result.replace("+", " + ")
            result = result.replace("-", " - ")
            result = result.strip()
        else:
            result = "0.0"
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.coefficients}, {self.symbols}, {self.exponents})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Polynomial) -> bool:
        return self.shape == other.shape and np.all(self.coefficients == other.coefficients) \
            and all(s == o for s, o in zip(self.symbols, other.symbols)) \
            and all(tuple(s) == tuple(o) for s, o in zip(self.exponents, other.exponents))


class PolarPolynomial(Polynomial):
    def __init__(self, coefficients: array_like, symbols: Sequence[str] = ('ρ', 'ϕ')):
        super().__init__(coefficients=coefficients, symbols=symbols)

    # def polar(self, rho: array_like, phi: array_like) -> array_type:
    #     return self(rho=rho, phi=phi)

    @property
    def gradient(self) -> Sequence[Polynomial]:
        grad = super().gradient  # Cartesian partial derivatives

        return grad[0], grad[1] * PolarPolynomial(1)
        # result: List[Polynomial] = list[Polynomial]()
        # for _, length in enumerate(self.shape):
        #     coefficients = self.coefficients.swapaxes(_, 0)[1:].swapaxes(0, _)
        #     exponents = np.arange(1, length)
        #     exponents = np.expand_dims(exponents, tuple(range(1, coefficients.ndim - _)))
        #     result.append(Polynomial(coefficients=coefficients * exponents, symbols=self.symbols))
        # return result


if __name__ == "__main__":
    # s = Polynomial(5)
    # print(repr(s))
    # print(s)

    p = Polynomial([[1, -4, 7], [5, 2, 3]], "xy")
    print(repr(p))
    print(p)

    p -= 1
    print(p)

    y = Polynomial([5], 'y', exponents=[[1]])
    print(f"y = {y}")

    p += y
    print(p)

    z = Polynomial([0, 0, 9], 'z', exponents=[[0, 10, 20]])
    print(f"z = {z}")

    p += z
    print(p)

    print(Polynomial([3.15], 't', exponents=[[-1.5]]))

    p = Polynomial([5, -3], 'x')
    print(repr(p))
    print(p)

    print(p([3.14128]))
