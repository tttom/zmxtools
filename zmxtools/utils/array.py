import warnings

import numpy as np

array_type = np.typing.NDArray[np.float16 | np.float32 | np.float64 | np.complex64 | np.complex128]
array_like = np.typing.ArrayLike


def asarray(_: array_like, dtype=np.complex64) -> array_type:
    """Converts numerical values to a NumPy ndarray of the desired type."""
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    arr = np.asarray(_, dtype=dtype)
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    return arr


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
