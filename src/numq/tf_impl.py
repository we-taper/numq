import numpy as np
import tensorflow as tf

from ._base_function import *

_tensor_types = [tf.Tensor, tf.SparseTensor, tf.Variable]


def _register_as(abs_func):
    def _reg(impl_func):
        """Micmic the behaviour of tf.contrib.framework.is_tensor(x) by register it for several
        tensorflow types. However, this migit not be perfect. See
        <https://www.tensorflow.org/api_docs/python/tf/contrib/framework/is_tensor> for tensorflow's
        implementation.
        """
        for t in _tensor_types:
            abs_func.register(t, impl_func)
        return impl_func

    return _reg


try:
    # tf v1
    # noinspection PyUnresolvedReferences
    _conj = tf.conj
except AttributeError:
    # tf v2
    _conj = tf.math.conj


@_register_as(apply_kraus_ops_to_density_matrices)
def apply_kraus_ops_to_density_matrices_tf(kraus_ops, density_matrices):
    # with tf.variable_scope("apply_kraus_ops_to_density_matrices_tf"):
    num_kraus_ops, matrix_dim, matrix_dim2 = kraus_ops.shape
    if matrix_dim != matrix_dim2:
        raise ValueError(kraus_ops.shape)
    num_wfs, den_mat_dim, den_mat_dim2 = density_matrices.shape
    if den_mat_dim != den_mat_dim2:
        raise ValueError(density_matrices.shape)
    if matrix_dim != den_mat_dim:
        raise ValueError("{0:d}, {1:d}".format(int(matrix_dim), int(den_mat_dim)))
    del matrix_dim2, den_mat_dim2, den_mat_dim

    mat_b = tf.tensordot(kraus_ops, density_matrices, axes=[[2], [1]])
    assert mat_b.shape == (num_kraus_ops, matrix_dim, num_wfs, matrix_dim)
    ret = tf.tensordot(mat_b, tf.linalg.adjoint(kraus_ops), axes=[[0, 3], [0, 1]])
    return tf.transpose(ret, perm=[1, 0, 2])


@_register_as(apply_unitary_transformation_to_density_matrices)
def apply_unitary_transformation_to_density_matrices_tf(unitary, density_matrices):
    # with tf.variable_scope("apply_unitary_transformation_to_density_matrices"):
    unitary = tf.convert_to_tensor(unitary)
    density_matrices = tf.convert_to_tensor(density_matrices)
    dim1, dim2 = unitary.shape
    num_states, dim3, dim4 = density_matrices.shape
    if dim1 != dim2 or dim2 != dim3 or dim3 != dim4 or dim1 != dim4:
        raise ValueError(
            "Shape error. Unitary:{0}, density:{1}.".format(
                str(unitary.shape), str(density_matrices.shape)
            )
        )

    # unitary U[i,k] D[nwf,k,l] -> B[i,nwf,l]
    mat_b = tf.tensordot(unitary, density_matrices, axes=[[1], [1]])
    assert mat_b.shape == (dim1, num_states, dim1)
    # B[i,nwf,k] U'[k,l] -> ret[i,nwf,l]
    ret = tf.tensordot(mat_b, tf.linalg.adjoint(unitary), axes=[[2], [0]])
    return tf.transpose(ret, perm=[1, 0, 2])


@_register_as(load_state_into_mqb_start_from_lqb)
def load_state_into_mqb_start_from_lqb_tf(states: tf.Tensor, m: int, l: int = 0):
    m = int(m)
    l = int(l)

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

    dtype_tf = states.dtype
    dtype_np = dtype_tf.as_numpy_dtype

    if h1dim == 1:
        h1states = np.ones(shape=(1, nwf), dtype=dtype_np)
    else:
        h1states = np.zeros(shape=(h1dim, nwf), dtype=dtype_np)
        h1states[0, :] = 1
    if h3dim == 1:
        h3states = np.ones(shape=(1, nwf), dtype=dtype_np)
    else:
        h3states = np.zeros(shape=(h3dim, nwf), dtype=dtype_np)
        h3states[0, :] = 1

    overall = [None] * nwf
    for wfidx in range(nwf):
        overall[wfidx] = kron_matrix_tf(
            kron_vector(h1states[:, wfidx], states[:, wfidx]), h3states[:, wfidx]
        )
    overall = tf.convert_to_tensor(overall, dtype=dtype_tf)
    overall = tf.transpose(tf.squeeze(overall))
    assert overall.shape == (big_h_dim, nwf)
    return overall


@_register_as(load1qb_into_ithqb)
def load1qb_into_ithqb_tf(iwf_1qb: tf.Tensor, target_qbidx: int, numphyqb: int):
    tfcpx_type = iwf_1qb.dtype
    n_wf = int(iwf_1qb.shape[1])
    numphyqb = int(numphyqb)
    target_qbidx = int(target_qbidx)
    owfs = [None] * n_wf
    if target_qbidx == 0:
        zeros = np.zeros((2 ** (numphyqb - 1), n_wf), dtype=tfcpx_type.as_numpy_dtype)
        zeros[0, :] = 1  # put it into 00000 state
        zeros = tf.convert_to_tensor(zeros, dtype=tfcpx_type)
        for wf_idx in range(n_wf):
            owfs[wf_idx] = kron_vector(iwf_1qb[:, wf_idx], zeros[:, wf_idx])
    elif target_qbidx == numphyqb - 1:
        zeros = np.zeros((2 ** (numphyqb - 1), n_wf), dtype=tfcpx_type.as_numpy_dtype)
        zeros[0, :] = 1  # put it into 00000 state
        zeros = tf.convert_to_tensor(zeros, dtype=tfcpx_type)
        for wf_idx in range(n_wf):
            owfs[wf_idx] = kron_vector(zeros[:, wf_idx], iwf_1qb[:, wf_idx])
    else:
        zeros_left = np.zeros(
            shape=(2 ** target_qbidx, n_wf), dtype=tfcpx_type.as_numpy_dtype
        )
        zeros_left[0, :] = 1
        zeros_right = np.zeros(
            shape=(2 ** (numphyqb - 1 - target_qbidx), n_wf),
            dtype=tfcpx_type.as_numpy_dtype,
        )
        zeros_right[0, :] = 1
        zeros_left = tf.convert_to_tensor(zeros_left, dtype=tfcpx_type)
        zeros_right = tf.convert_to_tensor(zeros_right, dtype=tfcpx_type)
        for wf_idx in range(n_wf):
            owfs[wf_idx] = kron_matrix_tf(
                kron_vector(zeros_left[:, wf_idx], iwf_1qb[:, wf_idx]),
                tf.expand_dims(zeros_right[:, wf_idx], axis=1),
            )
    # owfs is now a list of wavefunctions
    owfs = tf.convert_to_tensor(owfs, dtype=tfcpx_type)
    owfs = owfs[:, :, 0]  # squeeze is because [ (8,1), (8,1) ] -> (2, 8, 1)
    assert owfs.shape == (n_wf, 2 ** numphyqb)
    return tf.transpose(owfs)


@_register_as(kron_each)
def kron_each_tf(wf1s, wf2s):
    nwf = int(wf1s.shape[1])  # int to convert TensorFlow dimension into integer
    if nwf != int(wf2s.shape[1]):
        raise ValueError("Inconsistent number of input wavefunctions.")
    dim1 = wf1s.shape[0]
    dim2 = wf2s.shape[0]
    # with tf.variable_scope('kron_each'):
    wf1s = tf.convert_to_tensor(wf1s)
    wf2s = tf.convert_to_tensor(wf2s)
    # assume wf1s.shape = (dim1, nwf), wf2s.shape = (dim2, nwf)
    # we want to produce out[n=j*k, i] = wf1s[j,i] * wf2s[k,i]
    # we do this by broadcasting
    wf1s = wf1s[:, tf.newaxis, :]  # dim1, dummy, nwf
    wf2s = wf2s[tf.newaxis, :, :]  # dummy, dim2, nwf
    ret = wf1s * wf2s  # broadcasted to (dim1, dim2, nwf)
    return tf.reshape(ret, shape=(dim1 * dim2, nwf))


def kron_matrix_tf(mat1, mat2):
    # TODO: register and standardise its behaviour, esp. w.r.t. to vectors.
    if mat1.shape == (1,):
        return mat1[0] * mat2
    elif mat2.shape == (1,):
        return mat1 * mat2[0]
    m1 = mat1.shape[0]
    m2 = mat2.shape[0]
    try:
        n1 = mat1.shape[1]
    except IndexError:
        n1 = 1
    try:
        n2 = mat2.shape[1]
    except IndexError:
        n2 = 1
    with tf.name_scope("kron_matrix"):
        mat1 = tf.convert_to_tensor(mat1)
        mat2 = tf.convert_to_tensor(mat2)
        # copied from:
        # https://github.com/tensorflow/kfac/blob/06a82ef1a5d2640a9e3ce63e300f71663e6d2054/kfac/python/ops/utils.py#L113
        mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
        mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
        return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def kron_vector(a, b):
    # TODO: register this
    return kron_matrix_tf(tf.expand_dims(a, axis=1), tf.expand_dims(b, axis=1))


@_register_as(make_density_matrix)
def make_density_matrix_tf(wf):
    # with tf.variable_scope('make_density_matrix'):
    # einsum('i,j->ij', u, v)
    if len(wf.shape) == 1:
        a_wf_left = tf.expand_dims(wf, axis=1)
        a_wf_right = _conj(tf.expand_dims(wf, axis=0))
        return a_wf_left * a_wf_right
    elif len(wf.shape) == 2:
        _, num_wf = wf.shape
        ret = []
        for wf_idx in range(num_wf):
            a_wf = wf[:, wf_idx]
            a_wf_left = tf.expand_dims(a_wf, axis=1)
            a_wf_right = _conj(tf.expand_dims(a_wf, axis=0))
            ret.append(a_wf_left * a_wf_right)
        return tf.convert_to_tensor(ret)
    else:
        raise NotImplementedError(wf.shape)


@_register_as(partial_trace_1d)
def partial_trace_1d_tf(rho, retain_qubit: int):
    total_qb = int(np.log2(int(rho.shape[0])))

    if retain_qubit >= total_qb or retain_qubit < 0:
        raise ValueError(retain_qubit)

    if total_qb == 1:
        return rho

    all_qbs = list(range(total_qb))
    qbs_to_remove = list(filter(lambda x: x != retain_qubit, all_qbs))
    assert qbs_to_remove == list(sorted(qbs_to_remove))

    # with tf.variable_scope('partial_trace_1d'):
    transpose = tf.transpose
    trace = tf.linalg.trace
    rho = tf.reshape(rho, shape=[2] * (2 * total_qb))

    def tf_trace(tensor, kill_qb):
        """Implements trace which removes `kill_qb`th qubit"""
        # E.g. kill_qb = 2, total_qb = 5
        num_idx = 2 * total_qb
        perm = list(range(num_idx))
        # perm = k0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        perm[kill_qb + total_qb], perm[-1] = perm[-1], perm[kill_qb + total_qb]
        # perm = k0, 1, 2, 3, 4, 5, 6, 9, 8, 7
        perm[kill_qb], perm[-2] = perm[-2], perm[kill_qb]
        # perm = k0, 1, 8, 3, 4, 5, 6, 9, 2, 7
        tensor = transpose(tensor, perm=perm)
        tensor = trace(tensor)
        # perm = k0, 1, 8, 3, 4, 5, 6, 9,
        perm2 = list(range(num_idx - 2))
        for idx in range(kill_qb, num_idx - 2 - 1):
            perm2[idx], perm2[idx + 1] = perm2[idx + 1], perm2[idx]
        # perm = k0, 1, 3, 4, 5, 6, 8, 9
        tensor = transpose(tensor, perm=perm2)
        return tensor

    for qid in reversed(qbs_to_remove):
        # remove the qubit with higher qubit count first, this is crucial otherwise we will
        # have indexing problems.
        rho = tf_trace(rho, kill_qb=qid)
        total_qb -= 1  # one qubit removed already.
    return rho


@_register_as(pure_state_overlap)
def pure_state_overlap_tf(wf1, wf2):
    # with tf.variable_scope("pure_state_overlap"):
    # assert isinstance(wf1, tf.Tensor)
    if wf1.shape != wf2.shape:
        raise ValueError(
            "Inconsistent shape:\nwf1:{0:s}\twf2:{1:s}".format(
                str(wf1.shape), str(wf2.shape)
            )
        )
    if len(wf1.shape) == 1:
        wf1 = _conj(wf1)
        return tf.reduce_sum(wf1 * wf2)
    if len(wf1.shape) == 2:
        wf1 = _conj(wf1)
        return tf.reduce_sum(wf1 * wf2, axis=0)

    raise ValueError(str(wf1.shape))


@_register_as(trace_distance)
def trace_distance_tf(target_rho, pred_rho):
    # with tf.variable_scope("trace_distance"):
    try:
        # tf v1
        # noinspection PyUnresolvedReferences
        return 0.5 * tf.reduce_sum(
            tf.abs(tf.self_adjoint_eig(target_rho - pred_rho)[0])
        )
    except AttributeError:
        # tf v2
        return 0.5 * tf.reduce_sum(tf.abs(tf.linalg.eigh(target_rho - pred_rho)[0]))


@_register_as(trace_distance_1qb)
def trace_distance_1qb_tf(target_rho, pred_rho):
    # with tf.variable_scope("trace_distance"):
    assert target_rho.shape == (2, 2)
    diff = target_rho - pred_rho
    a = diff[0, 0]
    b = diff[0, 1]
    c = diff[1, 1]
    b_conj = diff[1, 0]
    return 0.25 * (
        tf.abs(a + c - tf.sqrt((a - c) ** 2 + 4 * b * b_conj))
        + tf.abs(a + c + tf.sqrt((a - c) ** 2 + 4 * b * b_conj))
    )


@_register_as(trace_distance_using_svd)
def trace_distance_using_svd_tf(target_rho, pred_rho):
    # with tf.variable_scope("trace_distance"):
    return 0.5 * tf.reduce_sum(
        tf.abs(tf.linalg.svd(target_rho - pred_rho, compute_uv=False))
    )


def tensorflow_optimised_config():
    # see this: https://www.tensorflow.org/guide/performance/overview#optimizing_for_cpu
    try:
        # noinspection PyUnresolvedReferences
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 6
        config.inter_op_parallelism_threads = 6
        return config
    except AttributeError:
        raise NotImplementedError("Not implemented in TF v2")


@_register_as(partial_trace_wf)
def partial_trace_wf_tf(iwf, retain_qubits):
    nqb = int(np.log2(int(iwf.shape[0])))
    iwf = tf.reshape(iwf, [2] * nqb)
    trace_out_indices = [i for i in range(0, nqb) if i not in retain_qubits]
    iwf_conj = tf.math.conj(iwf)
    out_shape0 = 2 ** len(retain_qubits)
    return tf.reshape(
        tf.tensordot(iwf, iwf_conj, axes=[trace_out_indices, trace_out_indices]),
        (out_shape0, out_shape0),
    )


@_register_as(partial_trace)
def partial_trace_tf(rho, retain_qubits):
    retain_qubits = sorted(list(retain_qubits))
    nqb = int(np.log2(int(rho.shape[0])))
    rho = tf.reshape(rho, [2] * (2 * nqb))
    address_to_trance_away = [x for x in range(nqb) if x not in retain_qubits]

    transpose_order = (
        retain_qubits
        + [x + nqb for x in retain_qubits]
        + address_to_trance_away
        + [x + nqb for x in address_to_trance_away]
    )

    transposed_density_matrix = tf.transpose(rho, transpose_order)

    transposed_density_matrix = tf.reshape(
        transposed_density_matrix,
        [2] * len(retain_qubits) * 2 + [2 ** len(address_to_trance_away)] * 2,
    )

    return_shape0 = 2 ** len(retain_qubits)
    return tf.reshape(
        tf.linalg.trace(transposed_density_matrix), (return_shape0, return_shape0)
    )
