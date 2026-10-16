from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

type NP_INT_TYPE = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
type NP_FLOAT_TYPE = np.float32 | np.float64
type NP_COMPLEX_TYPE = np.complex64 | np.complex128
type INT_TYPE = int | NP_INT_TYPE
type FLOAT_TYPE = float | NP_FLOAT_TYPE
type COMPLEX_TYPE = complex | NP_COMPLEX_TYPE
type SCALAR_TYPE = INT_TYPE | FLOAT_TYPE | COMPLEX_TYPE
SCALAR_TYPEVAR = TypeVar("SCALAR_TYPEVAR", NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)

type array_type[T: np.generic] = npt.NDArray[T]
type _nested_sequence[T] = T | npt.NDArray[Any] | Sequence["_nested_sequence[T]"]
type array_like[T: np.generic] = SCALAR_TYPE | npt.NDArray[Any] | _nested_sequence[SCALAR_TYPE]

asarray = np.asarray


def concatenate[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](*args: array_like[T]) -> array_type[T]:
    """Stacks values into a (higher dimensional) array _without_ adding a dimension on the right."""
    return np.concatenate(tuple(asarray(_) for _ in args), axis=-1)


def stack[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](*args: array_like[T]) -> array_type[T]:
    """Stacks values into a (higher dimensional) array by adding a dimension on the right."""
    return np.stack(tuple(asarray(_) for _ in args), axis=-1)


def dot[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](a: array_like[T], b: array_like[T]) -> array_type[T]:
    """Multiplies two arrays and sums their elements along the right-most dimension."""
    return einsum("...i,...i->...", a, b)


def cross[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](a: array_like[T], b: array_like[T]) -> array_type[T]:
    """Computes the cross product of two arrays along the right-most dimension."""
    return np.cross(asarray(a), asarray(b))


def einsum[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](subscripts: str, *args: array_like[T]) -> array_type[T]:
    """Computes the tensor product using the einstein summation convention."""
    return np.einsum(subscripts, *tuple(asarray(_) for _ in args))


def norm[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](_: array_like[T]) -> array_type[T]:
    """Computes the l2-norm along the right-most axis."""
    return np.linalg.norm(asarray(_), axis=-1)


def to_length[T: (NP_INT_TYPE, NP_FLOAT_TYPE, NP_COMPLEX_TYPE)](
    vector: array_like[T], length: int, value=0
) -> array_type[T]:
    """
    Pads a 1D vector to a given length, crops it if it is too long.

    Args:
        vector: The input vector.
        length: The length of the output vector.
        value: The value to use for padding.

    Returns:
        An output vector of the specified length in which the first values coincide and the rest is filled up with the provided value.

    """
    vector = asarray(vector)
    result = np.empty(shape=(length,), dtype=vector.dtype)
    copy_length = min(vector.size, length)
    result[:copy_length] = vector.ravel()[:copy_length]
    result[vector.size :] = value

    return result
