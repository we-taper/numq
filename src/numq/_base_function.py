"""
Declared base functions for q_toolkit module.

All functions provided by q_toolkit should be first declared here, which gives them a default
implementation (i.e. raising errors). After declared here, those functions can be overloaded in
their own module package to provide specific implementations.

For a specific implementation, it needs to register itself first inside
"mlec/q_toolkit/_register.py" before it can be used.

Note that when declaring default implementations, the default associated type is `object`. This
gives the following behaviour:

If the function is called, and the arguments CAN NOT be considered as a sub-type of any overloaded,
then the error provided by the default function (declared here) is raised.


"""
from functools import singledispatch
from typing import Iterable, List

__all__ = [
    "apply_isometry_to_density_matrices",
    "apply_kraus_ops_to_density_matrices",
    "apply_unitary_transformation_to_density_matrices",
    "commutator",
    "dagger",
    "distance",
    "equiv",
    "format_wavefunction",
    "get_random_ru",
    "get_randu",
    "get_random_wf",
    "rand_rho",
    "rand_herm",
    "kron_each",
    "load1qb_into_ithqb",
    "load_state_into_mqb_start_from_lqb",
    "load_states_into",
    "_load_states_into",
    "make_density_matrix",
    "partial_trace",
    "partial_trace_1d",
    "partial_trace_wf",
    "partial_trace_wf_keep_first",
    "pure_state_overlap",
    "trace_distance",
    "trace_distance_1qb",
    "trace_distance_using_svd",
]

_ERROR_MSG = (
    "There is not implementation registered for {type}."
    "Please ensure you have loaded the corresponding module first."
)


def _get_error_message(*args):
    types = str(list(map(str, map(type, args))))
    return _ERROR_MSG.format(type=types)


@singledispatch
def apply_isometry_to_density_matrices(isometry, density_matrices):
    """Applies the isometry to the density matrices.
    
    :param isometry: The isometry to apply.
    :param density_matrices: The density matrices to which the isometry is applied.
    :return: The density matrices after applying the isometry.
    """
    raise NotImplementedError(_get_error_message(isometry, density_matrices))


@singledispatch
def apply_kraus_ops_to_density_matrices(kraus_ops, density_matrices):
    """Applies the Kraus operators to the density matrices.

    :param kraus_ops: The Kraus operators to apply.
    :param density_matrices: The density matrices to which the Kraus operators are applied.
    :return: The density matrices after applying the Kraus operators.
    """
    raise NotImplementedError(_get_error_message(kraus_ops, density_matrices))


@singledispatch
def apply_unitary_transformation_to_density_matrices(unitary, density_matrices):
    """Applies the unitary transformation to the density matrices.

    :param unitary: The unitary transformation to apply.
    :param density_matrices: The density matrices to which the unitary transformation is applied.
    :return: The density matrices after applying the unitary transformation.
    """
    raise NotImplementedError(_get_error_message(unitary, density_matrices))


@singledispatch
def commutator(a, b):
    """Returns the commutator of two matrices or operators.

    :param a: The first matrix or operator.
    :param b: The second matrix or operator.
    :return: The commutator [a, b] = ab - ba.
    """
    raise NotImplementedError(_get_error_message(a, b))


@singledispatch
def dagger(mat):
    raise NotImplementedError(_get_error_message(mat))


@singledispatch
def distance(mat1, mat2):
    raise NotImplementedError(_get_error_message(mat1, mat2))


@singledispatch
def equiv(mat1, mat2, **allclose_args):
    raise NotImplementedError(_get_error_message(mat1, mat2))


@singledispatch
def format_wavefunction(wf, precision=8, skip_zeros=False, reverse_qubitstr=False):
    """Prints the components of the wavefunction visually.
    The argument reverse_qubitstr will reverse the sting representation of qubit's basis.
    For example, wf[1] will go from the |01> basis to the |10> basis.

    :param wf: The wavefunction to format.
    :param precision: The number of decimal places to display.
    :param skip_zeros: If True, skips the components that are close to zero.
    :param reverse_qubitstr: If True, reverses the string representation of qubit's basis. This is similar to reversing the endianness in binary representation.
    :return: A formatted string representation of the wavefunction.
    """
    raise NotImplementedError(_get_error_message(wf))


def get_random_ru(np, n=2):
    """Get a random unitary matrix of size n x n through QR decomposition of a random matrix with real and imaginary parts each sampled element-wise from a uniform distribution.

    :param np: The X-py module to use (NumPy or CuPy).
    :param n: The size of the unitary matrix to generate.
    :return: A random unitary matrix of size n x n.
    """

    rr = np.random.rand(n, n)
    ri = np.random.rand(n, n)
    rc = rr + 1j * ri
    q, r = np.linalg.qr(rc)
    diag = np.diag
    r = diag(diag(r) / np.abs(diag(r)))
    return q.dot(r)


def get_random_wf(np, nqb, nwf=1):
    """Get a random wavefunction distributed according to Haar measure on the complex unitary sphere.

    :param np: The X-py module to use (NumPy or CuPy).
    :param nqb: The number of qubits.
    :param nwf: The number of wavefunctions to generate. Default is 1.

    :return: A random wavefunction of one-dimension or an array of random wavefunctions of two dimensions, with the second (i.e. last) dimension indexing the wavefunction.
    """
    if nwf == 1:
        wf = _get_random_wf_1(np, nqb)
    else:
        wf = np.empty(shape=(2 ** nqb, nwf), dtype=np.complex128)
        for wfidx in range(nwf):
            wf[:, wfidx] = _get_random_wf_1(np, nqb)
    return wf


def _get_random_wf_1(np, nqb):
    u = get_randu(np, nqb)
    p = u.dot(np.ones(shape=(2 ** nqb, 1)))
    p /= np.linalg.norm(p, ord=2)
    p = np.squeeze(p)
    return p


def get_randu(np, nqb):
    """
    Generates a n qubit random unitary matrix, distributed uniformly
    according to the Haar measure. This is based on http://www.dr-qubit.org/matlab/randU.m.

    :param np: The X-py module to use (NumPy or CuPy).
    :param nqb: The number of qubits.
    :return: A random unitary matrix of size 2^nqb x 2^nqb.
    """
    randn = np.random.randn
    diag = np.diag
    dim = 2 ** nqb
    x = randn(dim, dim) + 1j * randn(dim, dim)
    x /= np.sqrt(2)
    q, r = np.linalg.qr(x, mode="complete")
    diag_r = diag(r)
    r = diag(diag_r / np.abs(diag_r))
    return q.dot(r)


def rand_rho(np, nqb=None, dim=None):
    p = 10 * rand_herm(np, nqb, dim)
    ppp = p.dot(np.conj(np.transpose(p)))
    return ppp / np.trace(ppp)


def rand_herm(np, nqb=None, dim=None):
    """
    Generates a random Hermitian matrix. Based on http://www.dr-qubit.org/matlab/randH.m .

    :param np: The X-py module to use (NumPy or CuPy).
    :param nqb: The number of qubits. If provided, will override the `dim` parameter.
    :param dim: The dimension of the Hermitian matrix. If not provided, it is set to 2^nqb.

    :return: A random Hermitian matrix of size dim x dim.
    """
    if nqb is None:
        if not isinstance(dim, int):
            raise ValueError(f"Wrong parameters: nqb={nqb}, dim={dim}")
    else:
        dim = 2 ** nqb
    randn = np.random.randn
    h = 2 * (randn(dim, dim) + 1j * randn(dim, dim)) - (1 + 1j)
    return h + np.conj(np.transpose(h))


def get_rho_from_random_wf(np, nqb):
    """Get a random density matrix rho, by generating a random wavefunction
    and return its density matrix."""
    wf = _get_random_wf_1(np, nqb)
    wf = wf.reshape(2 ** nqb, 1)
    return np.kron(wf, wf.transpose().conj())


@singledispatch
def kron_each(wf1s, wf2s):
    raise NotImplementedError(_get_error_message(wf1s, wf2s))


@singledispatch
def load1qb_into_ithqb(iwf_1qb, target_qbidx: int, numphyqb: int):
    raise NotImplementedError(_get_error_message(iwf_1qb, target_qbidx, numphyqb))


@singledispatch
def load_state_into_mqb_start_from_lqb(states, m, l: int = 0):
    """
    Loads states (columns of wavefunctions) from the smaller, and initialise the new
    qubits to 0.

    Note: l is 0-based. m is 1-based. For example, you can load state into 1 qb starts
    from 0 qubit.

    :param states: The input states to load.
    :param m: The number of qubits in the target state.
    :param l: The starting qubit index (0-based) from which to load the states.

    :return: The states loaded into the specified qubits.
    """
    raise NotImplementedError(_get_error_message(states, m, l))


def load_states_into(states, total_qb: int, pos: List[int]):
    """
    Loads the input `states` into the positions (`pos`), within a total
    qubits by `total_qb`.

    :param states: The input states to load, which should be 
    :param total_qb: The total number of qubits in the system.
    :param pos: The positions (0-based) in which to load the states.

    :return: The states loaded into the specified positions.
    """
    assert total_qb > 0, "Illegal total_qb."
    assert all(0 <= _i < total_qb for _i in pos), "pos outside total_qb"
    assert 2 ** len(pos) == states.shape[0], "Input wf dim >= len(pos)"
    return _load_states_into(states, total_qb, pos)


@singledispatch
def _load_states_into(states, total_qb: int, pos: List[int]):
    raise NotImplementedError(_get_error_message(states, total_qb, pos))


@singledispatch
def make_density_matrix(wf):
    """Make a density matrix from a wavefunction, i.e., compute the "ket-bra".
    
    :param wf: The wavefunction to convert into a density matrix.
    :return: The density matrix corresponding to the wavefunction.
    """
    raise NotImplementedError(_get_error_message(wf))


@singledispatch
def partial_trace(rho, retain_qubits: Iterable[int]):
    """
    Compute the partial trace of rho.

    :param rho: The density matrix to perform the partial trace on.
    :param return_qubits: the qubits which we want to keep after partial trace.

    :return: The reduced density matrix after performing the partial trace.
    """
    raise NotImplementedError(_get_error_message(rho, retain_qubits))


@singledispatch
def partial_trace_1d(rho, retain_qubit: int):
    raise NotImplementedError(_get_error_message(rho, retain_qubit))


@singledispatch
def partial_trace_wf(iwf, retain_qubits: Iterable[int]):
    """Partial trace on wavefunctions. See `partial_trace` for an explanation.
    """
    raise NotImplementedError(_get_error_message(iwf, retain_qubits))


@singledispatch
def partial_trace_wf_keep_first(iwf, n: int):
    """Partial trace which keeps only the first n qubits

    :param iwf: The input wavefunction to perform the partial trace on.
    :param n: The number of qubits to keep after the partial trace. If n is 0, it returns a shape (1,1) array of value 1.
    
    Notes
    --------
    We keep the least significant bit on the left. That is, inside the bit
    string of :math:`n_0n_1\\cdots n_m`, :math:`n_0` is the first qubit.

    In this way, for a iwf of indices
    :math:`i_1\\cdots i_n j_{n+1}j_{n+2}\\cdots`, the :math:`j` indices
    needs to be summed, which should be close together on the memory since we
    assume that iwf is c-ordered (although the code would work even if iwf is
    f-ordered)

    """
    raise NotImplementedError(_get_error_message(iwf, n))


@singledispatch
def pure_state_overlap(wf1, wf2):
    raise NotImplementedError(_get_error_message(wf1, wf2))


@singledispatch
def trace_distance(target_rho, pred_rho):
    raise NotImplementedError(_get_error_message(target_rho, pred_rho))


@singledispatch
def trace_distance_1qb(target_rho, pred_rho):
    raise NotImplementedError(_get_error_message(target_rho, pred_rho))


@singledispatch
def trace_distance_using_svd(target_rho, pred_rho):
    raise NotImplementedError(_get_error_message(target_rho, pred_rho))
