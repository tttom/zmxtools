import numpy as np

from zmxtools.utils.array import to_length, array_like, array_type


def factorial_fraction(numerator: array_like = 0, denominator: array_like = 0) -> array_type:
    """
    Calculates the quotient of two factorials, or arrays of factorials, attempting to avoid overflows.

    :param numerator: An integer or array of integers.
    :param denominator: An integer or array of integers.
    :return: A number or array of numbers of the same shape as the inputs.
    """
    numerator = np.array(numerator)
    denominator = np.array(denominator)
    difference = np.array(numerator - denominator)
    data_shape = difference.shape

    result = np.ones(shape=data_shape, dtype=float)

    for idx in np.arange(2, 1 + np.maximum(np.amax(numerator), np.amax(denominator))):
        # Iterate both the numerator and the denominator
        num_bool = np.logical_and(denominator < idx, idx <= numerator)  # either 0 or 1 for every element
        den_bool = np.logical_and(numerator < idx, idx <= denominator)  # either 0 or 1, but never both 1
        # either 1/idx, 1, or idx for every element
        result[num_bool] *= idx
        result[den_bool] *= 1 / idx

    return result.reshape(data_shape)


def factorial_product_fraction(numerators: array_like | tuple[array_like] = (),
                               denominators: array_like | tuple[array_like] = (),
                               ) -> array_type:
    """
    Calculates the quotient of two products of factorials, or arrays of factorials, attempting to avoid overflows.

    If either input argument is not a tuple, it is wrapped in one.

    :param numerators: A set of integers or arrays of integers.
    :param denominators: A set of integers or arrays of integers.

    :return: A number or array of numbers of the same shape as the inputs.
    """
    if not isinstance(numerators, tuple):
        numerators = (numerators, )
    if not isinstance(denominators, tuple):
        denominators = (denominators, )

    max_numerator = 1
    data_shape = np.array((), dtype=np.uint32)
    for n in numerators:
        n = np.asarray(n)
        if n.size > 0:
            max_numerator = np.maximum(max_numerator, np.max(n))
            # Expand data_shape so it encompasses all arguments
            if n.ndim > data_shape.size:
                data_shape = to_length(data_shape, n.ndim, 0)
            data_shape = np.maximum(data_shape, np.array(n.shape, dtype=int))
    max_denominator = 1
    for d in denominators:
        d = np.asarray(d)
        if d.size > 0:
            max_denominator = np.maximum(max_denominator, np.amax(d))
            # Expand data_shape so it encompasses all arguments
            if d.ndim > data_shape.size:
                data_shape = to_length(data_shape, d.ndim, 0)
            data_shape = np.maximum(data_shape, np.array(d.shape, dtype=int))

    # Check if we should better do this as the inverse fraction and revert it at the end
    inverse_calculation = max_denominator > max_numerator
    if inverse_calculation:
        numerators, denominators = denominators, numerators
        max_numerator, max_denominator = max_denominator, max_numerator

    # Do the calculation starting from all 2! factors
    result = np.ones(shape=data_shape, dtype=float)

    # Multiply only the factors that don't cancel on both sides of the fraction
    for idx in np.arange(2, 1 + np.maximum(max_numerator, max_denominator)):
        # Iterate both the numerator and the denominator
        numerator_idx_factors = sum((idx <= np.asarray(_)) for _ in numerators)
        numerator_idx_factors -= sum((idx <= np.asarray(_)) for _ in numerators)
        result *= np.array(idx, dtype=float) ** numerator_idx_factors

    result = result.reshape(data_shape)

    if inverse_calculation:
        result = 1.0 / result

    return result
