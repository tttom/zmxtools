import warnings

import numpy as np

any_numeric_dtype = (np.uint8 | np.uint16 | np.uint32 | np.uint64 |
                     np.int8 | np.int16 | np.int32 | np.int64 |
                     np.float16 | np.float32 | np.float64 | np.float128 |
                     np.complex64 | np.complex128 | np.complex256
                     )
array_type = np.typing.NDArray[any_numeric_dtype]
array_like = np.typing.ArrayLike


def asarray(_: array_like, dtype=np.complex64) -> array_type:
    """Converts numerical values to a NumPy ndarray of the desired type."""
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    arr = np.asarray(_, dtype=dtype)
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    return arr


def stack(*args: array_like) -> array_type:
    """Stacks values into a (higher dimensional) array by adding a dimension on the right."""
    return np.stack(args, axis=-1)


def dot(a: array_like, b: array_like) -> array_type:
    """Multiplies two arrays and sums their elements along the right-most dimension."""
    return einsum('...i,...i->...i', a, b)


def cross(a: array_like, b: array_like) -> array_type:
    """Computes the cross product of two arrays along the right-most dimension."""
    return np.cross(a, asarray(b), axis=-1)


def einsum(subscripts: str, *args: array_like) -> array_type:
    """Computes the tensor product using the einstein summation convention."""
    return np.einsum(subscripts, *(asarray(arg) for arg in args))


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
