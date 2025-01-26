import warnings
from typing import Sequence, TypeAlias

import numpy as np

array_type = np.ndarray
array_like: TypeAlias = array_type | int | float | complex | Sequence['array_like']


def asarray(_: array_like, dtype=np.complex64) -> array_type:
    """Converts numerical values to a NumPy ndarray of the desired type."""
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    _ = np.asarray(_, dtype=dtype)
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    return _

def stack(*args: array_like) -> array_type:
    """Stacks values into a (higher dimensional) array."""
    return np.stack(*args, axis=-1)


def dot(a: array_like, b: array_like) -> array_type:
    """Multiplies two arrays and sums their elements."""
    return np.dot(a, b)


def cross(a: array_like, b: array_like) -> array_type:
    """Computes the cross product of two arrays."""
    return np.cross(a, b)


def einsum(subscripts: str, *args: array_like) -> array_type:
    """Computes the tensor product using the einstein summation convention."""
    return np.einsum(subscripts, *args)


def norm(_: array_like) -> array_type:
    """Computes the l2-norm along the right-most axis."""
    return np.linalg.norm(_, axis=-1)


def to_length(vector, length: int, value=0):
    """
    Pads a 1D vector to a given length, crops it if it is too long.

    :param vector: The input vector.
    :param length: The length of the output vector.
    :param value: The value to use for padding.

    :return: An output vector of the specified length in which the first values coincide and the rest is filled up with
        the provided value.
    """
    vector = np.array(vector)
    result = np.empty(shape=(length, ), dtype=vector.dtype)
    copy_length = min(vector.size, length)
    result[:copy_length] = vector.ravel()[:copy_length]
    result[vector.size:] = value

    return result
