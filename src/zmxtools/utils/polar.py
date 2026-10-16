"""
A module to convert numpy arrays between Cartesian and polar coordinates.
"""

import numpy as np

from zmxtools.utils.array import array_like, array_type, asarray


def cart2pol(
    y: array_like,
    x: array_like,
) -> tuple[array_type, array_type]:
    """
    Convert Cartesian coordinates to polar coordinates.

    phi == 0 corresponds to the x-axis of the right-hand coordinate.
    phi == pi/2 corresponds to the y-axis of the left-hand coordinate.

    Args:
        y: The vertical (left-hand) dimension.
        x: The horizontal (right-hand) dimension.

    Returns:
        A tuple, (rho, phi), with the azimuthal and radial coordinate, respectively.

    """
    x_arr = asarray(x)
    y_arr = asarray(y)
    rho = np.hypot(x_arr, y_arr)
    phi = np.arctan2(y_arr, x_arr)

    return rho, phi


def pol2cart(
    rho: array_like,
    phi: array_like,
) -> tuple[array_type, array_type]:
    """
    Convert polar coordinates to Cartesian coordinates.

    phi == 0 corresponds to the x-axis of the right-hand coordinate.
    phi == pi/2 corresponds to the y-axis of the left-hand coordinate.

    Args:
        rho: The radial coordinate.
        phi: The azimuthal coordinate.

    Returns:
        A tuple, (y, x), with the respective Cartesian coordinates.

    """
    rho = asarray(rho)
    phi = asarray(phi)
    return rho * np.sin(phi), rho * np.cos(phi)
