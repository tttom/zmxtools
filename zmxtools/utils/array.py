from typing import Sequence, TypeAlias
import numpy as np

array_type: TypeAlias = np.ndarray
array_like: TypeAlias = array_type | int | float | complex | Sequence["array_like"]


def asarray(_: array_like, dtype=np.complex64) -> array_type:
    return np.asarray(_, dtype=dtype)


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
