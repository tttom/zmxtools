from __future__ import annotations

import warnings
from typing import Iterator, Protocol, TypeAlias, TypeVar, no_type_check, runtime_checkable

import numpy as np

INTEGER_TYPE = TypeVar('INTEGER_TYPE',
                       np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64,
                       covariant=True,
                       )
SCALAR_TYPE = TypeVar('SCALAR_TYPE',
                      np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64,
                      np.float16, np.float32, np.float64,
                      np.complex64 | np.complex128 | np.complex256,
                      covariant=True,
                      )


@runtime_checkable
class _SupportsArray(Protocol[SCALAR_TYPE]):
    def __array__(self) -> np.ndarray[tuple[int, ...], np.dtype[SCALAR_TYPE]]:
        pass


@runtime_checkable
class _NestedSequence(Protocol[SCALAR_TYPE]):
    """A protocol for representing nested sequences."""

    def __len__(self, /) -> int:
        """Implement ``len(self)``."""
        raise NotImplementedError

    def __getitem__(self, index: int, /) -> SCALAR_TYPE | _NestedSequence[SCALAR_TYPE]:
        """Implement ``self[x]``."""
        raise NotImplementedError

    def __contains__(self, x: object, /) -> bool:
        """Implement ``x in self``."""
        raise NotImplementedError

    def __iter__(self, /) -> Iterator[SCALAR_TYPE | _NestedSequence[SCALAR_TYPE]]:
        """Implement ``iter(self)``."""
        raise NotImplementedError

    def __reversed__(self, /) -> Iterator[SCALAR_TYPE | _NestedSequence[SCALAR_TYPE]]:
        """Implement ``reversed(self)``."""
        raise NotImplementedError

    def count(self, value, /) -> int:
        """Return the number of occurrences of `value`."""
        raise NotImplementedError

    def index(self, value, /) -> int:
        """Return the first index of `value`."""
        raise NotImplementedError


array_type: TypeAlias = np.typing.NDArray[SCALAR_TYPE]
array_like: TypeAlias = (array_type[SCALAR_TYPE] | SCALAR_TYPE | int | float | complex |
                         _SupportsArray[SCALAR_TYPE] | _NestedSequence[SCALAR_TYPE]
                         )


def asarray(_: array_like[SCALAR_TYPE], /, dtype=np.complex64) -> array_type[SCALAR_TYPE]:
    """Converts numerical values to a NumPy ndarray of the desired type."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
        arr = np.asarray(_, dtype=dtype)
    return arr


def concatenate(*args: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
    """Stacks values into a (higher dimensional) array _without_ adding a dimension on the right."""
    return np.concatenate(tuple(asarray(_) for _ in args), axis=-1)


def stack(*args: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
    """Stacks values into a (higher dimensional) array by adding a dimension on the right."""
    return np.stack(tuple(asarray(_) for _ in args), axis=-1)


def dot(a: array_like[SCALAR_TYPE], b: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
    """Multiplies two arrays and sums their elements along the right-most dimension."""
    return einsum('...i,...i->...i', a, b)


@no_type_check
def cross(a: array_like[SCALAR_TYPE], b: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
    """Computes the cross product of two arrays along the right-most dimension."""
    return np.cross(a, b)


def einsum(subscripts: str, *args: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
    """Computes the tensor product using the einstein summation convention."""
    return np.einsum(subscripts, *tuple(asarray(_) for _ in args))


def norm(_: array_like[SCALAR_TYPE]) -> array_type[SCALAR_TYPE]:
    """Computes the l2-norm along the right-most axis."""
    return np.linalg.norm(asarray(_), axis=-1)


def to_length(vector: array_like[SCALAR_TYPE], length: int, value=0) -> array_type[SCALAR_TYPE]:
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
