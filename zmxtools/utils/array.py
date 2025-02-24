from __future__ import annotations

from typing import TypeVar

import numpy as np

NP_INT_TYPE = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
NP_FLOAT_TYPE = np.float32 | np.float64
NP_COMPLEX_TYPE = np.complex64 | np.complex128
INT_TYPE = int | NP_INT_TYPE
FLOAT_TYPE = float | NP_FLOAT_TYPE
COMPLEX_TYPE = complex | NP_COMPLEX_TYPE
SCALAR_TYPEVAR = TypeVar('SCALAR_TYPEVAR', NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)
array_type = np.typing.NDArray[SCALAR_TYPEVAR]
array_like = INT_TYPE | FLOAT_TYPE | COMPLEX_TYPE | np.typing.ArrayLike | array_type[SCALAR_TYPEVAR]

asarray = np.asarray


def concatenate(*args: array_like[SCALAR_TYPEVAR]) -> array_type[SCALAR_TYPEVAR]:
    """Stacks values into a (higher dimensional) array _without_ adding a dimension on the right."""
    return np.concatenate(tuple(asarray(_) for _ in args), axis=-1)


def stack(*args: array_like[SCALAR_TYPEVAR]) -> array_type[SCALAR_TYPEVAR]:
    """Stacks values into a (higher dimensional) array by adding a dimension on the right."""
    return np.stack(tuple(asarray(_) for _ in args), axis=-1)


def dot(a: array_like[SCALAR_TYPEVAR], b: array_like[SCALAR_TYPEVAR]) -> array_type[SCALAR_TYPEVAR]:
    """Multiplies two arrays and sums their elements along the right-most dimension."""
    return einsum('...i,...i->...i', a, b)


def cross(a: array_like[SCALAR_TYPEVAR], b: array_like[SCALAR_TYPEVAR]) -> array_type[SCALAR_TYPEVAR]:
    """Computes the cross product of two arrays along the right-most dimension."""
    return np.cross(a, b)  # type: ignore


def einsum(subscripts: str, *args: array_like[SCALAR_TYPEVAR]) -> array_type[SCALAR_TYPEVAR]:
    """Computes the tensor product using the einstein summation convention."""
    return np.einsum(subscripts, *tuple(asarray(_) for _ in args))


def norm(_: array_like[SCALAR_TYPEVAR]) -> array_type[SCALAR_TYPEVAR]:
    """Computes the l2-norm along the right-most axis."""
    return np.linalg.norm(asarray(_), axis=-1)


def to_length(vector: array_like[SCALAR_TYPEVAR], length: int, value=0) -> array_type[SCALAR_TYPEVAR]:
    """
    Pads a 1D vector to a given length, crops it if it is too long.

    :param vector: The input vector.
    :param length: The length of the output vector.
    :param value: The value to use for padding.

    :return: An output vector of the specified length in which the first values coincide and the rest is filled up with
        the provided value.
    """
    vector = asarray(vector)
    result = np.empty(shape=(length, ), dtype=vector.dtype)
    copy_length = min(vector.size, length)
    result[:copy_length] = vector.ravel()[:copy_length]
    result[vector.size:] = value

    return result
