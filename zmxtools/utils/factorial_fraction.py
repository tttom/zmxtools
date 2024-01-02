import numpy as np

from zmxtools.utils import to_length


def factorial_fraction(numerator=0, denominator=0):
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
    # if result.ndim < 1:
    #     result = result[np.newaxis]

    for idx in np.arange(2, 1 + np.maximum(np.max(numerator), np.max(denominator))):
        # Iterate both the numerator and the denominator
        num_bool = (denominator < idx) & (idx <= numerator)  # either 0 or 1 for every element
        den_bool = (numerator < idx) & (idx <= denominator)  # either 0 or 1, but never both 1
        # either 1/idx, 1, or idx for every element
        result[num_bool] *= idx
        result[den_bool] *= 1 / idx

    return result.reshape(data_shape)


def factorial_product_fraction(numerators: tuple=(), denominators: tuple=()):
    """
    Calculates the quotient of two products of factorials, or arrays of factorials, attempting to avoid overflows.
    If either input argument is not a tuple, it is wrapped in one.

    :param numerators: A set of integers or arrays of integers.
    :param denominators: A set of integers or arrays of integers.
    :return: An number or array of numbers of the same shape as the inputs.
    """
    if not isinstance(numerators, tuple):
        numerators = (numerators, )
    if not isinstance(denominators, tuple):
        denominators = (denominators, )

    max_numerator = 1
    data_shape = np.array((), dtype=np.uint32)
    for n in numerators:
        n = np.array(n)
        if n.size > 0:
            max_numerator = np.maximum(max_numerator, np.max(n))
            # Expand data_shape so it encompasses all arguments
            if n.ndim > data_shape.size:
                data_shape = to_length(data_shape, n.ndim, 0)
            data_shape = np.maximum(data_shape, np.array(n.shape, dtype=int))
    max_denominator = 1
    for n in denominators:
        n = np.array(n)
        if n.size > 0:
            max_denominator = np.maximum(max_denominator, np.max(n))
            # Expand data_shape so it encompasses all arguments
            if n.ndim > data_shape.size:
                data_shape = to_length(data_shape, n.ndim, 0)
            data_shape = np.maximum(data_shape, np.array(n.shape, dtype=int))

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
        numerator_idx_factors = np.zeros(data_shape, dtype=np.int32)
        # if numerator_idx_factors.ndim < 1:
        #     numerator_idx_factors = numerator_idx_factors[..., np.newaxis]
        for n in numerators:
            numerator_idx_factors += idx <= np.array(n)  # either 0 or 1 for every element
        for n in denominators:
            numerator_idx_factors -= idx <= np.array(n)  # either 0 or 1 for every element

        # tmp = np.array(idx, dtype=float32)**numerator_idx_factors
        # log.debug(f"Shapes of result {result.shape}, tmp {tmp.shape}, and product {(result*tmp).shape}")

        result *= np.array(idx, dtype=float)**numerator_idx_factors

    result = result.reshape(data_shape)

    if inverse_calculation:
        result = 1.0 / result

    return result
