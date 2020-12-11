"""Numpy implementation of base functions."""

import numba
from numpy import (
    abs,
    allclose,
    array2string,
    asarray,
    conj,
    einsum,
    empty,
    empty_like,
    eye,
    kron,
    log2,
    moveaxis,
    ndarray,
    ones,
    outer,
    sqrt,
    squeeze,
    sum,
    swapaxes,
    tensordot,
    trace,
    transpose,
    zeros,
)
from numpy.linalg import eigvalsh, norm, svd

from ._base_function import *
from ._base_function import _load_states_into


@apply_kraus_ops_to_density_matrices.register(ndarray)
def apply_kraus_ops_to_density_matrices_numpy(kraus_ops, density_matrices):
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
    # mat_c = np.einsum('abik,akj->bij', mat_b, adjoint)
    mat_c = tensordot(mat_b, adjoint, axes=[[0, 3], [0, 1]])
    assert mat_c.shape == (num_wfs, matrix_dim, matrix_dim)
    return mat_c


@apply_kraus_ops_to_density_matrices.register(ndarray)
def apply_unitary_transformation_to_density_matrices_numpy(unitary, density_matrices):
    dim1, dim2 = unitary.shape
    num_states, dim3, dim4 = density_matrices.shape
    if dim1 != dim2 or dim2 != dim3 or dim3 != dim4 or dim1 != dim4:
        raise ValueError(
            "Shape error. Unitary:{0}, density:{1}.".format(
                str(unitary.shape), str(density_matrices.shape)
            )
        )

    # unitary U[i,k] D[nwf,k,l] -> B[i,nwf,l]
    mat_b = tensordot(unitary, density_matrices, axes=[[1], [1]])
    # B[i,nwf,k] U'[k,l] -> ret[i,nwf,l]
    ret = tensordot(mat_b, transpose(conj(unitary)), axes=[[2], [0]])
    return swapaxes(ret, axis1=0, axis2=1)


@commutator.register(ndarray)
def commutator_numpy(a: ndarray, b: ndarray):
    return a.dot(b) - b.dot(a)


@dagger.register(ndarray)
def dagger_numpy(mat: ndarray):
    return conj(transpose(mat))


@distance.register(ndarray)
def distance_numpy(mat1, mat2):
    tmp = dagger(mat1).dot(mat2)
    eye_alpha = tmp[0, 0] * eye(tmp.shape[0], dtype=tmp.dtype)
    return norm(tmp - eye_alpha, ord=2)


@equiv.register(ndarray)
def equiv_numpy(mat1, mat2, **allclose_args):
    tmp = dagger(mat1).dot(mat2)
    eye_alpha = tmp[0, 0] * eye(tmp.shape[0], dtype=tmp.dtype)
    return allclose(tmp, eye_alpha, **allclose_args)


@format_wavefunction.register(ndarray)
def format_wavefunction_numpy(
    wf, precision=8, skip_zeros=False, reverse_qubitstr=False
):
    wf = squeeze(asarray(wf))
    n = int(log2(wf.shape[0]))
    max_state = 2 ** n

    format_str_float = "{0:>" + str(precision + 3) + "." + str(precision) + "f}"

    def formatter(x):
        return (
            f"{format_str_float.format(x.real)}+" f"{format_str_float.format(x.imag)}i"
        )

    format_str = "{0:0" + str(n) + "b}"
    ret = []
    skip_zero_atol = 10 ** (-precision)
    for st in range(0, max_state):
        amp = wf[st]
        if skip_zeros and allclose(amp, 0, atol=skip_zero_atol):
            continue
        if reverse_qubitstr:
            s = "{0:s} |{1:s}>".format(
                array2string(
                    amp,
                    precision=precision,
                    suppress_small=True,
                    formatter={"complexfloat": formatter},
                ),
                format_str.format(st)[::-1],
                # "{0:0nb}".format(st) = bitset<n>(st)
            )
        else:
            s = "{0:s} |{1:s}>".format(
                array2string(
                    amp,
                    precision=precision,
                    suppress_small=True,
                    formatter={"complexfloat": formatter},
                ),
                format_str.format(st),  # "{0:0nb}".format(st) = bitset<n>(st)
            )
        ret.append(s)
    return "\n".join(ret)


@kron_each.register(ndarray)
def kron_each_numpy(wf1s, wf2s):
    nwf = int(wf1s.shape[1])  # int to convert TensorFlow dimension into integer
    if nwf != int(wf2s.shape[1]):
        raise ValueError("Inconsistent number of input wavefunctions.")
    owfs = empty(shape=(wf1s.shape[0] * wf2s.shape[0], wf1s.shape[1]), dtype=wf1s.dtype)
    for idx in range(wf1s.shape[1]):
        owfs[:, idx] = kron(wf1s[:, idx], wf2s[:, idx])
    return owfs


@load_state_into_mqb_start_from_lqb.register(ndarray)
def load_state_into_mqb_start_from_lqb_numpy(states: ndarray, m: int, l: int = 0):
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


@_load_states_into.register(ndarray)
def _load_states_into_numpy(states, total_qb, pos: tuple):
    states = load_state_into_mqb_start_from_lqb_numpy(states, total_qb, l=0)
    nwf = states.shape[1]
    states = states.reshape([2] * total_qb + [nwf])
    source = list(range(len(pos)))
    dest = list(pos)
    states = moveaxis(states, source, dest)
    states = states.reshape((2 ** total_qb, nwf), order="C")
    return states


@make_density_matrix.register(ndarray)
def make_density_matrix_numpy(wf: ndarray):
    if wf.ndim == 1:
        return outer(wf, wf.conjugate())
    if wf.ndim == 2:
        wf_dim, num_wf = wf.shape
        ret = empty(shape=(num_wf, wf_dim, wf_dim), dtype=wf.dtype)
        for wf_idx in range(num_wf):
            a_wf = wf[:, wf_idx]
            ret[wf_idx, :, :] = outer(a_wf, a_wf.conjugate())
        return ret
    raise NotImplementedError(wf.shape)


@partial_trace.register(ndarray)
def partial_trace_numpy(rho: ndarray, retain_qubits) -> ndarray:
    if len(retain_qubits) == 0:
        return trace(rho)

    total_qb = int(log2(rho.shape[0]))

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


@partial_trace_1d.register(ndarray)
def partial_trace_1d_numpy(rho, retain_qubit: int):
    total_qb = int(log2(int(rho.shape[0])))

    if retain_qubit >= total_qb or retain_qubit < 0:
        raise ValueError(retain_qubit)

    if total_qb == 1:
        return rho

    all_qbs = list(range(total_qb))
    qbs_to_remove = list(filter(lambda x: x != retain_qubit, all_qbs))
    assert qbs_to_remove == list(sorted(qbs_to_remove))
    rho = rho.reshape([2] * (2 * total_qb))
    # ret = empty(shape=(2,2), dtype=complex)
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


@partial_trace_wf.register(ndarray)
def partial_trace_wf_numpy(iwf: ndarray, retain_qubits):
    nqb = int(log2(iwf.shape[0]))
    iwf = iwf.reshape([2] * nqb)
    trace_out_indices = [i for i in range(0, nqb) if i not in retain_qubits]
    iwf_conj = conj(iwf)
    out_shape0 = 2 ** len(retain_qubits)
    return tensordot(
        iwf, iwf_conj, axes=[trace_out_indices, trace_out_indices]
    ).reshape((out_shape0, out_shape0))


@partial_trace_wf_keep_first.register(ndarray)
@numba.njit(parallel=True)
def partial_trace_wf_keep_first_numpy(iwf: ndarray, n: int):  # pragma: no cover
    r"""
    Notes
    -----
    Basic principles. Let `l` be the number of qubits, `m` be `n+1`, then

    .. math::

        \rho_{i_1\cdots i_{n}, j_1 \cdots j_{n}} =
        \sum_{k_m\cdots k_l}
        \psi_{i_1,\cdots i_{n}, k_m, \cdots k_l}
        \psi^*_{j_1,\cdots j_{n}, k_m, \cdots k_l}


    """
    # pre-calculate the conjugate of iwf
    iwf_conj = iwf.conj()

    # how many indices are there for the first n qubits
    n_idx = 2 ** n
    # how many qubits
    nqb = int(log2(iwf.shape[0]))
    # how many are traced out
    m = nqb - n
    # the range of indices we need to trace out
    # Note that when n=0, m_idx = iwf.shape[0]
    m_idx = 2 ** m

    # pre allocate the return array.
    # Note that when n = 0, ret is a (1,1) array.
    ret = zeros(shape=(n_idx, n_idx), dtype=iwf.dtype)
    for i in numba.prange(n_idx):
        for j in numba.prange(n_idx):
            # now binary shift the i and j such in binary they are
            # i00..., j00...   (00... = [0] * m)
            # Note that when n=0, i==j==0, the shift produces
            # also i_shift == j_shift == 0
            i_shift = i << m
            j_shift = j << m
            for k in numba.prange(m_idx):
                # note i_shift + k produces, in binary, ik
                # since the last m part of i are 00...,
                # and k is inside this 00... .
                # Similar for j.
                ret[i, j] += iwf[i_shift + k] * iwf_conj[j_shift + k]
    return ret


@pure_state_overlap.register(ndarray)
def pure_state_overlap_numpy(wf1: ndarray, wf2: ndarray):
    if wf1.ndim != wf2.ndim:
        raise ValueError("wf1:{0:s}\nwf2:{1:s}".format(str(wf1.ndim), str(wf2.ndim)))
    if wf1.ndim == 1:
        return transpose(conj(wf1)).dot(wf2)
    if wf1.ndim == 2:
        wf1 = conj(wf1)
        return sum(wf1 * wf2, axis=0)  # element-wise
    raise ValueError(str(wf1.shape))


@trace_distance.register(ndarray)
def trace_distance_numpy(target_rho: ndarray, pred_rho: ndarray):
    return 0.5 * sum(abs(eigvalsh(target_rho - pred_rho)))


@trace_distance_1qb.register(ndarray)
def trace_distance_1qb_cupy(target_rho: ndarray, pred_rho: ndarray) -> ndarray:
    diff = target_rho - pred_rho
    a = diff[0, 0]
    b = diff[0, 1]
    c = diff[1, 1]
    b_conj = diff[1, 0]
    return 0.25 * (
        abs(a + c - sqrt((a - c) ** 2 + 4 * b * b_conj))
        + abs(a + c + sqrt((a - c) ** 2 + 4 * b * b_conj))
    )


@trace_distance_using_svd.register(ndarray)
def trace_distance_using_svd_numpy(target_rho, pred_rho):
    return 0.5 * sum(abs(svd(target_rho - pred_rho)[1]))
