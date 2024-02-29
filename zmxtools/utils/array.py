from typing import Sequence, TypeAlias
import numpy as np

array_type = np.ndarray
array_like: TypeAlias = array_type | int | float | complex | Sequence["array_like"]


def asarray(_: array_like, dtype=np.complex64) -> array_type:
    return np.asarray(_, dtype=dtype)


def stack(*args: array_like) -> array_type:
    return np.stack(*args, axis=-1)


def dot(a: array_like, b: array_like) -> array_type:
    return np.dot(a, b)


def cross(a: array_like, b: array_like) -> array_type:
    return np.cross(a, b)


def einsum(subscripts: str, *args: array_like) -> array_type:
    return np.einsum(subscripts, *args)


def norm(_: array_like) -> array_type:
    return np.linalg.norm(_, axis=-1)


def norm2(_: array_like) -> array_type:
    return norm(_) ** 2


def sin(_: array_like) -> array_type:
    return np.sin(_)


def cos(_: array_like) -> array_type:
    return np.cos(_)


def arctan2(a: array_like, b: array_like) -> array_type:
    return np.arctan2(a, b)


def sqrt(_: array_like) -> array_type:
    return np.sqrt(_)


def maximum(a: array_like, b: array_like) -> array_type:
    return np.maximum(a, b)


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
