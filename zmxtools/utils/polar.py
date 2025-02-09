"""
A module to convert :ph:class:``numpy.ndarray``s between Cartesian and polar coordinates.
"""
import numpy as np

from zmxtools.utils.array import SCALAR_TYPE, array_like, array_type, asarray


def cart2pol(y: array_like[SCALAR_TYPE], x: array_like[SCALAR_TYPE],
             ) -> tuple[array_type[SCALAR_TYPE], array_type[SCALAR_TYPE]]:
    """
    Convert Cartesian coordinates to polar coordinates.

    phi == 0 corresponds to the x-axis of the right-hand coordinate.
    phi == pi/2 corresponds to the y-axis of the left-hand coordinate.

    :param y: The vertical (left-hand) dimension.
    :param x: The horizontal (right-hand) dimension.
    :return: A tuple, (rho, phi), with the azimuthal and radial coordinate, respectively.
    """
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    return rho, phi


def pol2cart(rho: array_like[SCALAR_TYPE], phi: array_like[SCALAR_TYPE],
             ) -> tuple[array_type[SCALAR_TYPE], array_type[SCALAR_TYPE]]:
    """
    Convert polar coordinates to Cartesian coordinates.

    phi == 0 corresponds to the x-axis of the right-hand coordinate.
    phi == pi/2 corresponds to the y-axis of the left-hand coordinate.

    :param rho: The radial coordinate.
    :param phi: The azimuthal coordinate.
    :return: A tuple, (y, x), with the respective Cartesian coordinates.
    """
    rho = asarray(rho)
    return rho * np.sin(phi), rho * np.cos(phi)
