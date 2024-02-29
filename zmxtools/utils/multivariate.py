from __future__ import annotations

from collections import defaultdict
import itertools
import numpy as np
from typing import Callable, Sequence, Dict, List

from zmxtools.utils.polar import cart2pol
from zmxtools.utils.array import array_like, asarray, array_type
from zmxtools.utils import script


__all__ = ["Polynomial"]


class Polynomial(Callable):
    """
    A class to represent Cartesian multivariate polynomials.

    Generic exponents are allowed, positive and negative, i.e. as a Laurent polynomial.
    https://en.wikipedia.org/wiki/Laurent_polynomial
    """
    def __init__(self,
                 coefficients: array_like,
                 labels: Sequence[str] = tuple[str](),
                 exponents: Sequence[Sequence[int | float | complex]] = tuple[Sequence[int | float | complex]]()):
        """
        Construct a multivariate polynomial object that can be evaluated at specific points or array's thereof.

        :param coefficients: The coefficients as a multi-dimensional array with the N-th dimension corresponding to the
            N-th independent coordinate.
        :param labels: The names or symbols of the independent variables in order. This is used to display or to select
            the arguments by name. By default x₀, x₁, x₂, x₃, ... is used.
        :param exponents: The optional exponents of the polynomial. By default these are 0, 1, 2, ...
        """
        self.__coefficients = None
        self.coefficients = coefficients
        self.__symbols = tuple[str]()
        self.labels = labels
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
    def labels(self) -> Sequence[str]:
        """
        The symbols that are used to represent this as a str. Their number must equal self.coefficients.ndim
        """
        return self.__symbols

    @labels.setter
    def labels(self, new_symbols: Sequence[str]):
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
                exponents[variable_index] = (
                    *exponents[variable_index],
                    *range(len(exponents[variable_index]), self.shape[variable_index])
                )
            elif len(exponents[variable_index]) > self.shape[variable_index]:
                raise ValueError(f"The number of exponents, {len(exponents[variable_index])}, for {self.labels[variable_index]} should match the number of coefficients, {self.shape[variable_index]}.")
            else:
                exponents[variable_index] = tuple(exponents[variable_index])
        for variable_index in range(len(exponents), self.ndim):  # Add default exponents for the remaining dimensions
            exponents.append(tuple(range(self.shape[variable_index])))
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

    def __call__(self, *args: array_like, **kwargs: array_like) -> array_type:
        """
        The evaluated value of this polynomial at the specified coordinates. These can be specified in order of the
        labels or by label name. The coordinates are broadcast as necessary. Keyword arguments override non-named arguments.

        :param args: The coordinates in order of the symbols.
        :param kwargs: (optional) The coordinates by their label.

        :return: The polynomial value for each argument coordinate. The result has a shape that is equal to the
        broadcasted dimensions of the arguments.
        """
        for s in kwargs:
            assert s in self.labels, f"Unknown coordinate symbol, {s}. Must be one of {self.labels}."

        # Convert arguments to standard form
        arg_dict: Dict[str, array_type] = defaultdict[str, array_type](float)  # Default to 0.0
        for symbol, arg in zip(self.labels, args):
            arg_dict[symbol] = asarray(arg)
        for symbol, arg in kwargs.items():
            arg_dict[symbol] = asarray(arg)

        arguments = [arg_dict[label] for label in self.labels]
        # assert len(arguments) == len(self.symbols), f"Expected exactly one argument for each of {self.symbols}, got {len(arguments)}."
        while len(arguments) < self.ndim:  # Assume that missing arguments are 0.
            arguments.append(asarray(0.0))

        calculation_axes = tuple(range(-self.coefficients.ndim, 0))  # The axes of the multi-variate polynomial

        def calc_product_rec(coordinates: array_type, exponents) -> array_type:
            coordinate = np.expand_dims(coordinates[0], axis=calculation_axes)
            exponents_for_this_axis = np.expand_dims(exponents[0], axis=tuple(range(-(len(exponents) - 1), 0)))
            result = coordinate ** exponents_for_this_axis
            if len(exponents) > 1:
                result = result * calc_product_rec(coordinates[1:], exponents[1:])
            return result

        return np.sum(self.coefficients * calc_product_rec(arguments, self.exponents), axis=calculation_axes)

    def grad(self) -> Sequence[Polynomial]:
        """
        Calculate the gradient of this ``Polynomial``. The partial derivatives are listed in the order of ``self.labels``.
        """
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
                                     labels=self.labels,
                                     exponents=derivative_exponents))
        return result

    def __add__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Returns the sum of two polynomials, or a this ``Polynomial`` and a constant value."""
        if not isinstance(other, Polynomial):
            other = self.__class__(other)  # A scalar

        coefficients = self.coefficients
        symbols = list(self.labels)
        exponents = list(self.exponents)
        other_coefficients = other.coefficients
        other_symbols = list(other.labels)
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

        return self.__class__(coefficients=coefficients, labels=symbols, exponents=exponents)

    def __radd__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Add this Polynomial to something on the left or the right."""
        if not isinstance(other, Polynomial):
            other = self.__class__(other)
        return other + self

    def __neg__(self) -> Polynomial:
        """Negate the values of this ``Polynomial``."""
        return self.__class__(-self.coefficients, labels=self.labels, exponents=self.exponents)

    def __sub__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Return the difference of this polynomial and another."""
        return self + (-other)

    def __rsub__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            other = self.__class__(other)
        return other - self

    def __mul__(self, other: Polynomial | int | float | complex) -> Polynomial:
        """Scale this ``Polynomial`` by a scalar or multiply it with another Polynomial."""
        if not isinstance(other, Polynomial):
            return self.__class__(coefficients=self.coefficients * other, labels=self.labels, exponents=self.exponents)
        else:
            # unique_other_labels = (_ for _ in other.labels if _ not in self.labels)
            # product_labels = list(*self.labels, *unique_other_labels)
            # product_coefficients = np.expand_dims(self.coefficients, axis=tuple(range(-len(unique_other_labels))))
            raise NotImplementedError

    def __rmul__(self, other: int | float | complex) -> Polynomial:
        return Polynomial(coefficients=other * self.coefficients, labels=self.labels, exponents=self.exponents)

    def __truediv__(self, other: Polynomial | int | float | complex) -> Polynomial:
        if not isinstance(other, Polynomial):
            return self * (1 / other)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        """Format this polynomial as a unicode string."""
        def format_factor(symbol: str, exponent: int | float | complex) -> str:
            result = str(symbol) if exponent != 0 else ""
            if exponent != 0 and exponent != 1:
                result += script.sup(exponent)
            return result

        def format_coefficient(coefficient: complex, product_str: str) -> str:
            if coefficient.real != 0 and coefficient.imag != 0:
                result = f"+({coefficient.real}{coefficient.imag:+})"
            elif coefficient.real != 0:  # coefficient.imag == 0
                if product_str == "" or abs(coefficient) != 1:
                    result = f"{coefficient.real:+}"
                else:
                    result = "+" if coefficient == 1 else "-"
            else:  # coefficient.real == 0 but coefficient.imag != 0
                if coefficient.imag == -1:
                    result = "-i"
                elif coefficient.imag == 1:
                    result = "+i"
                else:
                    result = f"{coefficient.imag:+}i"

            result = result.replace("+", " + ")
            result = result.replace("-", " - ")

            return result

        products = itertools.product(*([format_factor(s, e) for e in self.exponents[_]] for _, s in enumerate(self.labels)))
        products = ("".join(_) for _ in products)
        terms = [format_coefficient(c, p) + "".join(p) for c, p in zip(self.coefficients.ravel(), products)
                 if c != 0]
        if len(terms) > 0:
            if terms[0].startswith(" +"):
                terms[0] = terms[0][2:]
            result = "".join(terms)
            result = result.strip()
        else:
            result = "0.0"
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.coefficients}, {self.labels}, {self.exponents})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Polynomial) -> bool:
        return self.shape == other.shape and np.all(self.coefficients == other.coefficients) \
            and all(s == o for s, o in zip(self.labels, other.labels)) \
            and all(tuple(s) == tuple(o) for s, o in zip(self.exponents, other.exponents))


class PolarPolynomial(Polynomial):
    def __init__(self, coefficients: array_like,
                 labels: Sequence[str] = ('ρ', 'ϕ'),
                 exponents: Sequence[Sequence[int | float | complex]] = tuple[Sequence[int | float | complex]]()):
        """
        A polynomial of the form Σₙᵐ aₙᵐ ρⁿ cos(mϕ) + Σₙᵐ bₙᵐ ρⁿ sin(mϕ)  =  Re Σₙᵐ cₙᵐ ρⁿ exp(-imϕ),
        with coefficients cₙᵐ = aₙᵐ + ibₙᵐ.

        :param coefficients: The coefficients as a multi-dimensional array with the N-th dimension corresponding to the
            N-th independent coordinate.
        :param labels: The names or symbols of the independent variables in order. This is used to display or to select
            the arguments by name. By default, the labels x₀, x₁, x₂, x₃, ... is used.
        :param exponents: The optional exponents of the polynomial. By default, these are 0, 1, 2, ...
        """
        cartesian_labels = (labels[0], f"exp(-i{labels[1]})")  # or "e⁻ⁱᵠ"
        super().__init__(coefficients=coefficients, labels=cartesian_labels, exponents=exponents)

    def __call__(self, rho: array_like = 0.0, phi: array_like = 0.0) -> array_type:
        """
        Calculates the ``PolarPolynomial``'s values at the specified polar coordinates. The coordinates are broadcast
        as necessary.

        :param rho: The radial coordinates.
        :param phi: The azimuthal coordinates in radians.

        :return: The values at the specified coordinates.
        """
        return super().__call__(rho, np.exp(-1j * asarray(phi)))

    def cartesian(self, y: array_like, x: array_like) -> array_type:
        """Compute the values at Cartesian coordinates. The coordinates are broadcast as necessary."""
        rho, phi = cart2pol(y, x)
        return self(rho, phi)

    def cartesian_grad(self, y: array_like, x: array_like, axis: int = 0) -> array_type:
        g = self.grad()

        rho, phi = cart2pol(y, x)
        epsilon = 1e-6
        at_origin = rho < epsilon  # At the origin phi should be 0

        phasors = np.exp(-1j * phi)
        df_drho, df_dphi = g[0](rho, phasors).real, (-1j * phasors * g[1](rho, phasors)).real

        # df_dphirho[at_origin] = (-1j * phasors * g[1].grad()[0](0.0, phasors)).real  # Apply l'Hopital
        df_dphirho = (1 - at_origin) * df_dphi / (rho + at_origin) \
                     + at_origin * (-1j * phasors * g[1].grad()[0](0.0, phasors)).real  # Apply l'Hopital if necessary
        c, s = np.cos(phi), np.sin(phi)
        df_dx = c * df_drho - s * df_dphirho
        df_dy = s * df_drho + c * df_dphirho

        return np.stack([df_dy, df_dx], axis=axis)


if __name__ == "__main__":
    from zmxtools.utils.polar import pol2cart
    import matplotlib.pyplot as plt

    # p = PolarPolynomial([[0.5 + 0.5j]], exponents=[[1], [1]])
    p = PolarPolynomial([[-2j * np.sqrt(8)], [3j * np.sqrt(8)]], exponents=[[1, 3], [1]]) / 5.0

    rho = np.arange(0, 1.1, 0.05)[..., np.newaxis]
    phi = np.arange(0, 1.05, 0.05) * 2 * np.pi
    y, x = pol2cart(rho, phi)

    z = p.cartesian(y, x).real
    grad = p.cartesian_grad(y, x)

    print("Calculating local coordinate system...")
    v = np.stack([grad[0] * 0 + 1, grad[0] * 0, grad[0]])
    u = np.stack([grad[1] * 0, grad[1] * 0 + 1, grad[1]])
    u /= np.linalg.norm(u, axis=0)
    v /= np.linalg.norm(v, axis=0)
    normal = np.cross(v, u, axis=0).real

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_aspect("equal")

    ax.plot_surface(x, y, z, color=(1, 1, 0.75), shade=True, linewidth=0, antialiased=False)
    ax.quiver(x, y, z, u[1], u[0], u[2], color=(0.75, 0, 0), length=0.1)
    ax.quiver(x, y, z, v[1], v[0], v[2], color=(0, 0.75, 0), length=0.1)
    ax.quiver(x, y, z, normal[1], normal[0], normal[2], color=(0, 0, 0.75), length=0.1)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="x", ylabel="y")
    plt.show()

