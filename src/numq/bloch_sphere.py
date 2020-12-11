"""
Utility functions related to bloch sphere.
"""
from collections import namedtuple
from typing import Union

import numpy as np

from .numpy_gates import x as X, y as Y, z as Z

CartCoord = namedtuple("CartCoord", ["x", "y", "z"])
PolarCoord = namedtuple("PolarCoord", ["r", "theta", "phi"])


def get_bloch_state(theta, phi):
    """Given a state's bloch angles theta and phi, returns the vector
    representing this bloch state.

    References
    -------------
    https://en.wikipedia.org/wiki/Bloch_sphere
    """
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0) * np.exp(1j * phi)])


def polar_to_cartesian(r=None, theta=None, phi=None, polar: PolarCoord = None):
    """Convert from Polar coordinate to Cartesian coordinate

    Using physicist's convention.

    References
    -----------
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Conventions
    """
    if polar is not None:
        r = polar.r
        theta = polar.theta
        phi = polar.phi
    rsintheta = r * np.sin(theta)
    x = rsintheta * np.cos(phi)
    y = rsintheta * np.sin(phi)
    z = r * np.cos(theta)
    return CartCoord(x, y, z)


def cartesian_to_polar(x=None, y=None, z=None, cart: CartCoord = None):
    """Convert from Cartesian coordinate to Polar coordinate

    Using physicist's convention.

    Returns
    --------
    r
    theta
    phi

    References
    -----------
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Conventions
    """
    if cart is not None:
        x = cart.x
        y = cart.y
        z = cart.z
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)
    return PolarCoord(r, theta, phi)


def bloch_denmatt(coord: Union[CartCoord, PolarCoord]):
    if isinstance(coord, PolarCoord):
        coord = polar_to_cartesian(polar=coord)
    if isinstance(coord, CartCoord):
        return (np.eye(2) + coord.x * X + coord.y * Y + coord.z * Z) * 0.5
    raise ValueError(type(coord))


def coord_of_denmat(rho: np.ndarray, ret_type=CartCoord):
    """Returns the Cartesian coordinates of a density matrix on a bloch sphere."""
    if rho.shape != (2, 2):
        raise NotImplementedError(rho.shape)
    x_cor = np.real(np.trace(rho.dot(X)))
    y_cor = np.real(np.trace(rho.dot(Y)))
    z_cor = np.real(np.trace(rho.dot(Z)))
    ret = CartCoord(x_cor, y_cor, z_cor)
    if ret_type == CartCoord:
        return ret
    elif ret_type == PolarCoord:
        return cartesian_to_polar(cart=ret)
