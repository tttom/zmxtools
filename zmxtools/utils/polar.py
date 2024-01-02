"""
A module to convert :ph:class:``numpy.ndarray``s between Cartesian and polar coordinates.
"""
import numpy as np


def cart2pol(y, x):
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


def pol2cart(rho, phi):
    """
    Convert polar coordinates to Cartesian coordinates.

    phi == 0 corresponds to the x-axis of the right-hand coordinate.
    phi == pi/2 corresponds to the y-axis of the left-hand coordinate.

    :param rho: The radial coordinate.
    :param phi: The azimuthal coordinate.
    :return: A tuple, (y, x), with the respective Cartesian coordinates.
    """
    y = rho * np.sin(phi)
    x = rho * np.cos(phi)

    return y, x
