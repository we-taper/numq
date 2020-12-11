import logging
import math
import os
from math import log2 as math_log2

import cupy
from cupy import (
    conj,
    einsum,
    empty,
    empty_like,
    kron,
    ones,
    outer,
    sqrt,
    swapaxes,
    tensordot,
    trace,
    transpose,
    zeros,
)
from cupy.cuda.memory import OutOfMemoryError
from cupy.linalg import eigvalsh, svd

from ._base_function import *
from .cupy_impl_cuda_source.wrapper import (
    default_dtype,
    partial_trace_wf_keep_first_cuda,
)

CUBLASError = cupy.cuda.cublas.CUBLASError
CUSOLVERError = cupy.cuda.cusolver.CUSOLVERError

logging.basicConfig()
logger = logging.getLogger(os.path.basename(__name__))

_abs = cupy.abs
_sum = cupy.sum

_tmpvec = cupy.array([[1.0], [1j]], dtype=complex) / cupy.sqrt(2)
_encountered_known_bug_in_cupy = 0

try:
    cupy.outer(_tmpvec, _tmpvec.conj())
except CUBLASError as e:
    _encountered_known_bug_in_cupy += 1
    cupy.outer(_tmpvec, _tmpvec.conj())

try:
    _tmprho = cupy.random.rand(2, 2) + 1j * cupy.random.rand(2, 2)
    _tmprho = _tmprho + cupy.conj(cupy.transpose(_tmprho))
    cupy.linalg.eigvalsh(_tmprho)
except CUSOLVERError as e:
    _encountered_known_bug_in_cupy += 1
    # noinspection PyUnboundLocalVariable
    cupy.linalg.eigvalsh(_tmprho)


@apply_isometry_to_density_matrices.register(cupy.ndarray)
def apply_isometry_to_density_matrices_cupy(
    isometry: cupy.ndarray, density_matrices: cupy.ndarray
):
    outdim, dim1 = isometry.shape
    num_states, dim2, dim3 = density_matrices.shape
    assert dim1 == dim2 == dim3

    # unitary U[i,k] D[nwf,k,l] -> B[i,nwf,l]
    mat_b = tensordot(isometry, density_matrices, axes=[[1], [1]])
    # B[i,nwf,k] U'[k,l] -> ret[i,nwf,l]
    ret = tensordot(mat_b, transpose(conj(isometry)), axes=[[2], [0]])
    return swapaxes(ret, axis1=0, axis2=1)


@apply_kraus_ops_to_density_matrices.register(cupy.ndarray)
def apply_kraus_ops_to_density_matrices_cupy(
    kraus_ops: cupy.ndarray, density_matrices: cupy.ndarray
):
    num_kraus_ops, matrix_dim, matrix_dim2 = kraus_ops.shape
    if matrix_dim != matrix_dim2:
        raise ValueError(kraus_ops.shape)
    num_wfs, den_mat_dim, den_mat_dim2 = density_matrices.shape
    if den_mat_dim != den_mat_dim2:
        raise ValueError(density_matrices.shape)
    if matrix_dim != den_mat_dim:
        raise ValueError("{0:d}, {1:d}".format(int(matrix_dim), int(den_mat_dim)))
    del matrix_dim2, den_mat_dim2, den_mat_dim

    mat_b = einsum("aij,bjk->abik", kraus_ops, density_matrices)
    assert mat_b.shape == (num_kraus_ops, num_wfs, matrix_dim, matrix_dim)
    adjoint = empty_like(kraus_ops)
    for idx in range(num_kraus_ops):
        adjoint[idx, :, :] = transpose(conj(kraus_ops[idx]))

    mat_c = tensordot(mat_b, adjoint, axes=[[0, 3], [0, 1]])
    assert mat_c.shape == (num_wfs, matrix_dim, matrix_dim)
    return mat_c


@apply_unitary_transformation_to_density_matrices.register(cupy.ndarray)
def apply_unitary_transformation_to_density_matrices_cupy(
    unitary: cupy.ndarray, density_matrices: cupy.ndarray
):
    dim1, dim2 = unitary.shape
    num_states, dim3, dim4 = density_matrices.shape
    assert dim1 == dim2 == dim3 == dim4
    # unitary U[i,k] D[nwf,k,l] -> B[i,nwf,l]
    mat_b = tensordot(unitary, density_matrices, axes=[[1], [1]])
    # B[i,nwf,k] U'[k,l] -> ret[i,nwf,l]
    ret = tensordot(mat_b, transpose(conj(unitary)), axes=[[2], [0]])
    return swapaxes(ret, axis1=0, axis2=1)


@format_wavefunction.register(cupy.ndarray)
def formwat_wavefunction_cupy(wf, *args, **kwargs):
    from .numpy_impl import format_wavefunction_numpy

    wf = wf.get()
    return format_wavefunction_numpy(wf, *args, **kwargs)


@kron_each.register(cupy.ndarray)
def kron_each_cupy(wf1s, wf2s):
    nwf = int(wf1s.shape[1])  # int to convert TensorFlow dimension into integer
    if nwf != int(wf2s.shape[1]):
        raise ValueError("Inconsistent number of input wavefunctions.")

    owfs = empty(shape=(wf1s.shape[0] * wf2s.shape[0], wf1s.shape[1]), dtype=wf1s.dtype)
    for idx in range(wf1s.shape[1]):
        owfs[:, idx] = kron(wf1s[:, idx], wf2s[:, idx])
    return owfs


@load_state_into_mqb_start_from_lqb.register(cupy.ndarray)
def load_state_into_mqb_start_from_lqb_cupy(states, m, l=0) -> cupy.ndarray:
    """
    Loads states (columns of wavefunctions) from the smaller, and initialise
     the new qubits to 0.
    """
    # performs parameter check
    assert m > l >= 0
    assert m >= 1

    # the Hilbert spaces are divided into three parts
    # h1: dim = 2^n1
    # h2: states', dim = h2dim = 2^m
    # h3: dim = 2^n2
    # 2^(n1 + m + n2) = big_h_dim
    h1dim = 2 ** l
    h2dim, nwf = states.shape
    big_h_dim = 2 ** m
    h3dim = big_h_dim // (h1dim * h2dim)
    assert h3dim >= 1

    dtype = states.dtype

    if h1dim == 1:
        h1states = ones(shape=(1, nwf), dtype=dtype)
    else:
        h1states = zeros(shape=(h1dim, nwf), dtype=dtype)
        h1states[0, :] = 1
    if h3dim == 1:
        h3states = ones(shape=(1, nwf), dtype=dtype)
    else:
        h3states = zeros(shape=(h3dim, nwf), dtype=dtype)
        h3states[0, :] = 1

    overall = empty(shape=(big_h_dim, nwf), dtype=dtype)
    for wfidx in range(nwf):
        overall[:, wfidx] = kron(
            kron(h1states[:, wfidx], states[:, wfidx]), h3states[:, wfidx]
        )
    return overall


@make_density_matrix.register(cupy.ndarray)
def make_density_matrix_cupy(wf: cupy.ndarray):
    if wf.ndim == 1:
        return outer(wf, cupy.conj(wf))
    if wf.ndim == 2:
        wf_dim, num_wf = wf.shape
        try:
            ret = empty(shape=(num_wf, wf_dim, wf_dim), dtype=wf.dtype)
        except OutOfMemoryError:
            logger.critical(
                "OOM when creating density matrix for wavefunction "
                f"of shape {wf.shape}"
            )
            raise
        for wf_idx in range(num_wf):
            a_wf = wf[:, wf_idx]
            ret[wf_idx, :, :] = outer(a_wf, conj(a_wf))
        return ret
    raise NotImplementedError(wf.shape)


@pure_state_overlap.register(cupy.ndarray)
def pure_state_overlap_cupy(wf1: cupy.ndarray, wf2: cupy.ndarray):
    """
    Returns:
          a complex scalar of array (depends on the shape of wf1 and wf2)
    """
    ndim = wf1.ndim
    if ndim != wf2.ndim:
        raise ValueError("wf1:{0:s}\nwf2:{1:s}".format(str(ndim), str(wf2.ndim)))
    if ndim == 1:
        return transpose(conj(wf1)).dot(wf2)
    elif ndim == 2:
        wf1 = conj(wf1)
        return _sum(wf1 * wf2, axis=0)  # element-wise
    else:
        raise ValueError(str(wf1.shape))


@partial_trace.register(cupy.ndarray)
def partial_trace_cupy(rho: cupy.ndarray, retain_qubits) -> cupy.ndarray:
    """
    Compute the partial trace of rho.
    Args:
        rho: input rho
        retain_qubits: the qubits which we want to keep after partial trace.
    """
    if len(retain_qubits) == 0:
        return trace(rho)

    total_qb = int(math.log2(rho.shape[0]))

    assert min(retain_qubits) >= 0 and max(retain_qubits) < total_qb

    if total_qb == 1 or len(retain_qubits) == total_qb:
        return rho
    all_qbs = list(range(total_qb))
    qbs_to_remove = list(filter(lambda x: x not in retain_qubits, all_qbs))
    rho = rho.reshape([2] * (2 * total_qb))
    for qid in reversed(qbs_to_remove):
        rho = trace(rho, axis1=qid, axis2=qid + total_qb)
        total_qb -= 1

    # retain back to normal density matrix
    newshape = 2 ** total_qb
    return rho.reshape(newshape, newshape)


@partial_trace_1d.register(cupy.ndarray)
def partial_trace_1d_cupy(rho: cupy.ndarray, retain_qubit: int):
    """
    Compute the partial trace of rho. Returns a reduced density matrix
     in the Hilbert space of "retain_qubit"th qubit.
    """
    total_qb = int(math.log2(rho.shape[0]))

    if retain_qubit >= total_qb or retain_qubit < 0:
        raise ValueError(retain_qubit)

    if total_qb == 1:
        return rho

    all_qbs = list(range(total_qb))
    qbs_to_remove = list(filter(lambda x: x != retain_qubit, all_qbs))
    assert qbs_to_remove == list(sorted(qbs_to_remove))
    rho = rho.reshape([2] * (2 * total_qb))
    # ret = np.empty(shape=(2,2), dtype=complex)
    ret = None
    for qid in reversed(qbs_to_remove):
        # remove the qubit with higher qubit count first, this is crucial
        # otherwise we will have indexing problems.
        if ret is None:
            ret = trace(rho, axis1=qid, axis2=qid + total_qb)
            total_qb -= 1  # removed one already
        else:
            ret = trace(ret, axis1=qid, axis2=qid + total_qb)
            total_qb -= 1  # removed one already
    assert ret.shape == (2, 2)
    return ret


@partial_trace_wf.register(cupy.ndarray)
def partial_trace_wf_cupy(iwf: cupy.ndarray, retain_qubits):
    nqb = int(math_log2(iwf.shape[0]))
    if len(retain_qubits) == nqb:
        return outer(iwf, iwf.conj())
    iwf = iwf.reshape([2] * nqb, order="C")
    retain_qubits = sorted(retain_qubits)
    for idx in range(len(retain_qubits)):
        r = retain_qubits[idx]
        if idx != r:
            iwf = iwf.swapaxes(idx, r)
    iwf = iwf.reshape((2 ** nqb,))
    return partial_trace_wf_keep_first_cupy(iwf, len(retain_qubits))


# noinspection PyPep8Naming
@partial_trace_wf_keep_first.register(cupy.ndarray)
def partial_trace_wf_keep_first_cupy(iwf: cupy.ndarray, n):
    # TODO: improve the cuda version of partial_trace_wf.
    #   For example, cleverly adjust the blockDim to deal with the other case
    assert iwf.flags.c_contiguous
    assert iwf.dtype == default_dtype
    iwf_conj = iwf.conj()
    nqb = int(math_log2(iwf.shape[0]))
    m = nqb - n
    m_idx = 2 ** m
    n_idx = 2 ** n

    rho = zeros(shape=(n_idx, n_idx), dtype=default_dtype, order="C")
    # Here we simply use the threadDim for i, j in the cuda code.
    threads_per_bloch = 32
    threadDim = (threads_per_bloch, threads_per_bloch)
    x = (n_idx + (threads_per_bloch - 1)) // threads_per_bloch
    blockDim = (x, x)
    partial_trace_wf_keep_first_cuda(
        grid=blockDim,
        block=threadDim,
        args=(
            iwf,
            iwf_conj,
            rho,
            m,
            m_idx,
            n_idx,
        ),
    )
    return rho


@trace_distance.register(cupy.ndarray)
def trace_distance_cupy(target_rho, pred_rho) -> cupy.ndarray:
    """Note: if the differences is slightly negative (e.g. -1e-7),
    the returned trace distance will contain nan values!
    """
    return 0.5 * _sum(_abs(eigvalsh(target_rho - pred_rho)))


@trace_distance_1qb.register(cupy.ndarray)
def trace_distance_1qb_cupy(
    target_rho: cupy.ndarray, pred_rho: cupy.ndarray
) -> cupy.ndarray:
    diff = target_rho - pred_rho
    a = diff[0, 0]
    b = diff[0, 1]
    c = diff[1, 1]
    b_conj = diff[1, 0]
    return 0.25 * (
        abs(a + c - sqrt((a - c) ** 2 + 4 * b * b_conj))
        + abs(a + c + sqrt((a - c) ** 2 + 4 * b * b_conj))
    )


@trace_distance_using_svd.register(cupy.ndarray)
def trace_distance_using_svd_cupy(target_rho, pred_rho) -> cupy.ndarray:
    return 0.5 * _sum(_abs(svd(target_rho - pred_rho)[1]))
