"""
Currently we have:
    X, Y, Z, I, H, S, T, ST, 
    CNOT, CZ
    CCZ
"""
from functools import reduce
from typing import Iterable

import numpy as np

__all__ = [
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "rphi",
    "t_gate",
    "s_gate",
    "cnot",
    "cz",
    "mat_dict",
    "fold",
]

mat_dict = dict()
np_kron = np.kron
_complex_dtype = np.complex128


def _register(k, g, check_unitary=True):
    """
    Notes
    -------
    The convention is that k is kept as all capital characters.

    """
    k = k.upper()
    if check_unitary:
        g_dag = np.conjugate(np.transpose(g))
        if not np.allclose(np.eye(g.shape[0]), g.dot(g_dag)):
            raise ValueError(f"The gate for key {k} is not unitary.")
    if k in mat_dict:
        raise KeyError(f"Key {k} has already been registered.")
    else:
        mat_dict[k] = g


def fold(strings: Iterable[str]):
    """Builds a matrix by reducing kron to all keys inside the strings

    Notes
    -------
    The convention is that all symbols are in upper case.
    """
    return reduce(np_kron, (mat_dict[k] for k in strings))


x = np.array([[0, 1], [1, 0]], dtype=_complex_dtype)
_register("X", x)
y = np.array([[0, -1j], [1j, 0]], dtype=_complex_dtype)
_register("Y", y)
z = np.array([[1, 0], [0, -1]], dtype=_complex_dtype)
_register("Z", z)
hadamard = np.array([[1, 1], [1, -1]], dtype=_complex_dtype) / np.sqrt(2)
_register("H", hadamard)
I = np.eye(2, dtype=_complex_dtype)
_register("I", I)
ccz = np.eye(8, dtype=_complex_dtype)
# 110 <-> 111
_110 = int("110", 2)
_111 = int("111", 2)
ccz[_111, _111] = -1
_register("CCZ", ccz)


def rx(angle):
    halfangle = angle / 2.0
    return np.array(
        [
            [np.cos(halfangle), -1j * np.sin(halfangle)],
            [-1j * np.sin(halfangle), np.cos(halfangle)],
        ],
        dtype=_complex_dtype,
    )


def ry(angle):
    halfangle = angle / 2.0
    return np.array(
        [
            [np.cos(halfangle), -np.sin(halfangle)],
            [np.sin(halfangle), np.cos(halfangle)],
        ],
        dtype=_complex_dtype,
    )


def rz(angle):
    halfangle = angle / 2.0
    return np.array(
        [[np.exp(-1j * halfangle), 0], [0, np.exp(1j * halfangle)]],
        dtype=_complex_dtype,
    )


def rphi(angle):
    return np.array([[1, 0], [0, np.exp(1j * angle)]])


t_gate = rphi(np.pi / 4.0)
_register("T", t_gate)
s_gate = rphi(np.pi / 2.0)
_register("S", s_gate)

cnot = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=_complex_dtype
)
_register("CNOT", cnot)
cz = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=_complex_dtype
)
_register("CZ", cz)

st = s_gate.dot(t_gate)
_register("ST", st)
