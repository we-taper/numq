from functools import reduce
from unittest import TestCase

import numpy as np
import pytest

from numq import (
    load_module,
    kron_each,
    get_random_wf,
    load_state_into_mqb_start_from_lqb,
    load_states_into,
    make_density_matrix,
    dagger,
    equiv,
    distance,
    pure_state_overlap,
    partial_trace_1d,
    partial_trace,
    partial_trace_wf,
    partial_trace_wf_keep_first,
    trace_distance,
    trace_distance_1qb,
    trace_distance_using_svd,
)
from numq.numpy_gates import *


def add_methods(cls):
    from types import MethodType

    cls.from_numpy = MethodType(lambda self, _: _, cls)
    cls.to_numpy = MethodType(lambda self, _: _, cls)

    # noinspection PyUnusedLocal
    def prepare_module(self):
        load_module("numpy")

    cls.prepare_module = MethodType(prepare_module, cls)

    return cls


@add_methods
class TestKronEach(TestCase):
    def setUp(self) -> None:
        self.prepare_module()

    def test_trivial(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        def kron_each_(a, b):
            return to_np(kron_each(from_np(a), from_np(b)))

        wf1s = np.identity(2)
        wf2s = np.identity(4)[:, 0:2]
        out = kron_each_(wf1s, wf2s)
        should_be = np.zeros(shape=(8, 2), dtype=float)
        should_be[0, 0] = 1
        should_be[5, 1] = 1
        assert np.allclose(out, should_be)

    def test_random_numpy(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        def kron_each_(a, b):
            return to_np(kron_each(from_np(a), from_np(b)))

        wf1s = np.random.rand(4, 3)
        wf2s = np.random.rand(3, 3)
        should_be = np.empty(shape=(12, 3), dtype=float)
        should_be[:, 0] = np.kron(wf1s[:, 0], wf2s[:, 0])
        should_be[:, 1] = np.kron(wf1s[:, 1], wf2s[:, 1])
        should_be[:, 2] = np.kron(wf1s[:, 2], wf2s[:, 2])
        out = kron_each_(wf1s, wf2s)
        assert np.allclose(out, should_be)


def _prep_load_state_data():
    state = get_random_wf(np, nqb=2, nwf=2)
    state0 = np.array([[1, 1], [0, 0]])
    state00 = np.array([[1, 1], [0, 0], [0, 0], [0, 0]])
    kron = np.kron
    loaded_into_01th = np.empty(shape=(2 ** 4, 2), dtype=complex)
    for wfidx in range(2):
        loaded_into_01th[:, wfidx] = kron(state[:, wfidx], state00[:, wfidx])
    loaded_into_12th = np.empty(shape=(2 ** 4, 2), dtype=complex)
    for wfidx in range(2):
        loaded_into_12th[:, wfidx] = kron(
            kron(state0[:, wfidx], state[:, wfidx]), state0[:, wfidx]
        )
    loaded_into_23th = np.empty(shape=(2 ** 4, 2), dtype=complex)
    for wfidx in range(2):
        loaded_into_23th[:, wfidx] = kron(state00[:, wfidx], state[:, wfidx])

    return state, loaded_into_01th, loaded_into_12th, loaded_into_23th


@add_methods
class TestLoadStatesToLargerHilbertSpace(TestCase):
    def setUp(self):
        tmp = _prep_load_state_data()
        self.state = tmp[0]
        self.loaded_into_01th = tmp[1]
        self.loaded_into_12th = tmp[2]
        self.loaded_into_23th = tmp[3]
        self.prepare_module()

    def test_accurately_loaded(self):
        from_numpy = self.from_numpy
        to_numpy = self.to_numpy
        state = from_numpy(self.state)
        wf1 = load_state_into_mqb_start_from_lqb(state, m=4, l=0)
        assert np.allclose(to_numpy(wf1), self.loaded_into_01th)
        wf1 = load_state_into_mqb_start_from_lqb(state, m=4, l=1)
        assert np.allclose(to_numpy(wf1), self.loaded_into_12th)
        wf1 = load_state_into_mqb_start_from_lqb(state, m=4, l=2)
        assert np.allclose(to_numpy(wf1), self.loaded_into_23th)

    def test_trivial_case(self):
        wf1 = load_state_into_mqb_start_from_lqb(self.from_numpy(self.state), m=2)
        assert np.array_equal(self.to_numpy(wf1), self.state)

    def test_raising_errors(self):
        state = self.from_numpy(self.state)
        with pytest.raises(AssertionError):
            load_state_into_mqb_start_from_lqb(state, m=1)
        with pytest.raises(AssertionError):
            load_state_into_mqb_start_from_lqb(state, m=3, l=2)


@add_methods
class TestLoadStatesInto(TestCase):
    def setUp(self):
        tmp = _prep_load_state_data()
        self.state = tmp[0]
        self.loaded_into_01th = tmp[1]
        self.loaded_into_12th = tmp[2]
        self.loaded_into_23th = tmp[3]

        self.prepare_module()

    def test_accurately_loaded(self):
        from_numpy = self.from_numpy
        to_numpy = self.to_numpy
        state = from_numpy(self.state)
        wf1 = load_states_into(state, total_qb=4, pos=[0, 1])
        assert np.allclose(to_numpy(wf1), self.loaded_into_01th)
        wf1 = load_states_into(state, total_qb=4, pos=[1, 2])
        assert np.allclose(to_numpy(wf1), self.loaded_into_12th)
        wf1 = load_states_into(state, total_qb=4, pos=[2, 3])
        assert np.allclose(to_numpy(wf1), self.loaded_into_23th)

    def test_trivial_case(self):
        wf1 = load_states_into(self.from_numpy(self.state), total_qb=2, pos=[0, 1])
        assert np.array_equal(self.to_numpy(wf1), self.state)

    def test_accurate_load_multi(self):
        from_numpy = self.from_numpy
        to_numpy = self.to_numpy
        wf13_1 = get_random_wf(np, nqb=1, nwf=2)
        wf13_2 = get_random_wf(np, nqb=1, nwf=2)
        wf13 = kron_each(wf13_1, wf13_2)
        state00 = np.zeros(shape=(2, 2), dtype=complex)
        state00[0, :] = 1.0
        loaded_into_13th_tot4qb = reduce(
            kron_each, [state00.copy(), wf13_1, state00.copy(), wf13_2]
        )

        wf13 = from_numpy(wf13)
        loaded = load_states_into(wf13, total_qb=4, pos=[1, 3])
        loaded = to_numpy(loaded)
        assert np.allclose(loaded, loaded_into_13th_tot4qb)

    def test_raising_errors(self):
        state = self.from_numpy(self.state)
        with pytest.raises(AssertionError):
            load_states_into(state, total_qb=3, pos=[0])
        with pytest.raises(AssertionError):
            load_states_into(state, total_qb=1, pos=[0, 1])
        with pytest.raises(AssertionError):
            load_states_into(state, total_qb=3, pos=[2, 3])


@add_methods
class TestMakeDensityMatrix(TestCase):
    def setUp(self) -> None:
        self.prepare_module()

        self.wf1 = np.array([[1], [1j]], dtype=complex) / np.sqrt(2)
        self.plus = np.array([[1], [1]]) / np.sqrt(2)
        self.minus = np.array([[1], [-1]]) / np.sqrt(2)
        self.rho1 = np.array([[1, -1j], [1j, 1]]) / 2.0
        self.rho_p = np.array([[1, 1], [1, 1]], dtype=complex) / 2.0
        self.rho_m = np.array([[1, -1], [-1, 1]], dtype=complex) / 2.0

    def test_make_density_mat_1d(self):
        from_np = self.from_numpy
        to_np = self.to_numpy
        rho1 = to_np(make_density_matrix(from_np(self.wf1.ravel())))
        assert np.allclose(rho1, self.rho1)
        rho_p = to_np(make_density_matrix(from_np(self.plus.ravel())))
        assert np.allclose(rho_p, self.rho_p)
        rho_m = to_np(make_density_matrix(from_np(self.minus.ravel())))
        assert np.allclose(rho_m, self.rho_m)

    def test_make_density_mat(self):
        from_np = self.from_numpy
        to_np = self.to_numpy
        wfs = np.hstack((self.wf1, self.plus, self.minus))
        rho = to_np(make_density_matrix(from_np(wfs)))
        should_be = np.asarray([self.rho1, self.rho_p, self.rho_m])
        assert np.allclose(rho, should_be)


@add_methods
class TestMisc(TestCase):
    def setUp(self) -> None:
        self.prepare_module()

    def test_dagger(self):
        randn = np.random.rand()
        a = np.exp(1j * randn) * np.eye(2)
        b = np.exp(-1j * randn) * np.eye(2)
        assert np.allclose(dagger(a), b)

    def test_equivalent_distance(self):
        a = np.eye(3)
        b = np.exp(1j * np.random.rand()) * a
        assert equiv(a, b)
        assert np.allclose(distance(a, b), 0)

    def test_rotation_gates(self):
        assert np.allclose(distance(rx(3 * np.pi), x), 0)
        assert np.allclose(distance(ry(3 * np.pi), y), 0)
        assert np.allclose(distance(rz(3 * np.pi), z), 0)
        assert equiv(rx(3 * np.pi), x)
        assert equiv(ry(3 * np.pi), y)
        assert equiv(rz(3 * np.pi), z)


@add_methods
class TestOverlap(TestCase):
    def setUp(self) -> None:
        self.prepare_module()

    def test(self):
        # TODO: move two functions to main lib:
        def _v_norm(wf: np.ndarray) -> float:
            """
            Returns the vector's 2-norm

            Parameters
            ----------
            wf :
                a vector

            Returns
            -------
            2-norm : float

            """
            return np.sqrt(np.sum(np.square(np.abs(wf))))

        def get_rand_wf(nqubit: int):
            """
            Return a vector whose real and complex entries are each picked from normal distribution::

                (mu, sigma = 0, 1),

            and is then normalised."""
            wf_real = np.random.normal(loc=0, scale=1, size=2 ** nqubit)
            wf_cpx = np.random.normal(loc=0, scale=1, size=2 ** nqubit) * 1j
            wf = wf_real + wf_cpx
            wf = wf / _v_norm(wf)
            return wf

        wfs1 = np.empty(shape=(2 ** 3, 4), dtype=complex)
        for i in range(4):
            wfs1[:, i] = get_rand_wf(3)
        wfs2 = np.empty(shape=(2 ** 3, 4), dtype=complex)
        for i in range(4):
            wfs2[:, i] = get_rand_wf(3)

        should_be = np.diag(np.transpose(np.conjugate(wfs1)).dot(wfs2))

        from_numpy = self.from_numpy
        to_numpy = self.to_numpy

        wfs1_tmp = from_numpy(wfs1)
        wfs2_tmp = from_numpy(wfs2)
        ret = to_numpy(pure_state_overlap(wfs1_tmp, wfs2_tmp))
        assert ret.shape == should_be.shape
        assert np.allclose(should_be, ret)

        should_be_single = np.transpose(np.conjugate(wfs1[:, 0])).dot(wfs2[:, 0])
        ret_single = pure_state_overlap(from_numpy(wfs1[:, 0]), from_numpy(wfs2[:, 0]))
        ret_single = to_numpy(ret_single)
        assert ret_single.shape == should_be_single.shape
        assert np.allclose(should_be_single, ret_single)


@add_methods
class TestPartialTrace(TestCase):
    def setUp(self) -> None:
        self.prepare_module()
        self.plus_minus = (
            np.array(
                [[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1],],
                dtype=complex,
            )
            / 4.0
        )
        self.plus = np.array([[1, 1], [1, 1],], dtype=complex) / 2.0
        self.minus = np.array([[1, -1], [-1, 1],], dtype=complex) / 2.0
        self.ppmm = np.kron(
            self.plus, np.kron(self.plus, np.kron(self.minus, self.minus))
        )
        self.mep = (
            np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0],], dtype=complex
            )
            / 2.0
        )
        self.mep_reduced = np.eye(2, dtype=complex) / 2.0

        self.rho_cpx = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        self.rho_cpx_0 = np.trace(
            self.rho_cpx.reshape([2] * 4), axis1=1, axis2=3
        )  # trace out the second system
        self.rho_cpx_1 = np.trace(
            self.rho_cpx.reshape([2] * 4), axis1=0, axis2=2
        )  # trace out the 1st system

    def test_work_for_pure_states_1d(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        plus_minus = self.plus_minus
        plus = self.plus
        minus = self.minus
        plus_to_test = partial_trace_1d(from_np(plus_minus), retain_qubit=0)
        minus_to_test = partial_trace_1d(from_np(plus_minus), retain_qubit=1)

        assert np.array_equal(plus, to_np(plus_to_test))
        assert np.array_equal(minus, to_np(minus_to_test))

        plus_plus_minus = self.ppmm
        p_1 = partial_trace_1d(from_np(plus_plus_minus), retain_qubit=0)
        p_2 = partial_trace_1d(from_np(plus_plus_minus), retain_qubit=1)
        m_1 = partial_trace_1d(from_np(plus_plus_minus), retain_qubit=2)
        m_2 = partial_trace_1d(from_np(plus_plus_minus), retain_qubit=3)

        assert np.array_equal(to_np(p_1), plus)
        assert np.array_equal(to_np(p_2), plus)
        assert np.array_equal(to_np(m_1), minus)
        assert np.array_equal(to_np(m_2), minus)

    def test_work_for_pure_states_ndim(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        plus_minus = self.plus_minus
        plus = self.plus
        minus = self.minus
        plus_to_test = partial_trace(from_np(plus_minus), retain_qubits=(0,))
        minus_to_test = partial_trace(from_np(plus_minus), retain_qubits=(1,))

        assert np.array_equal(plus, to_np(plus_to_test))
        assert np.array_equal(minus, to_np(minus_to_test))

        plus_plus_minus = self.ppmm
        p_1 = partial_trace(from_np(plus_plus_minus), retain_qubits=(0,))
        p_2 = partial_trace(from_np(plus_plus_minus), retain_qubits=(1,))
        m_1 = partial_trace(from_np(plus_plus_minus), retain_qubits=(2,))
        m_2 = partial_trace(from_np(plus_plus_minus), retain_qubits=(3,))

        assert np.array_equal(to_np(p_1), plus)
        assert np.array_equal(to_np(p_2), plus)
        assert np.array_equal(to_np(m_1), minus)
        assert np.array_equal(to_np(m_2), minus)

    def test_work_for_mixed_case_1d(self):
        mep = self.mep
        mep_reduced = self.mep_reduced
        from_np = self.from_numpy
        to_np = self.to_numpy
        rho1 = to_np(partial_trace_1d(from_np(mep), 0))
        rho2 = to_np(partial_trace_1d(from_np(mep), 1))
        assert np.array_equal(rho1, mep_reduced)
        assert np.array_equal(rho2, mep_reduced)

        rho_cpx = self.rho_cpx
        rho_0 = self.rho_cpx_0
        rho_1 = self.rho_cpx_1
        assert np.array_equal(rho_0, to_np(partial_trace_1d(from_np(rho_cpx), 0)))
        assert np.array_equal(rho_1, to_np(partial_trace_1d(from_np(rho_cpx), 1)))

    def test_work_for_mixed_case_ndim(self):
        mep = self.mep
        mep_reduced = self.mep_reduced
        from_np = self.from_numpy
        to_np = self.to_numpy
        rho1 = to_np(partial_trace(from_np(mep), (0,)))
        rho2 = to_np(partial_trace(from_np(mep), (1,)))
        assert np.array_equal(rho1, mep_reduced)
        assert np.array_equal(rho2, mep_reduced)

        rho_cpx = self.rho_cpx
        rho_0 = self.rho_cpx_0
        rho_1 = self.rho_cpx_1
        assert np.array_equal(rho_0, to_np(partial_trace(from_np(rho_cpx), (0,))))
        assert np.array_equal(rho_1, to_np(partial_trace(from_np(rho_cpx), (1,))))

    def test_trivial_case_1d(self):
        rand = np.random.rand
        from_np = self.from_numpy
        to_np = self.to_numpy

        rho = rand(2, 2) + 1j * rand(2, 2)
        rho_test = to_np(partial_trace_1d(from_np(rho), 0))
        assert np.array_equal(rho, rho_test)

    def test_trivial_case_ndim(self):
        rand = np.random.rand
        from_np = self.from_numpy
        to_np = self.to_numpy
        rho = rand(2, 2) + 1j * rand(2, 2)

        assert np.array_equal(
            rho, to_np(partial_trace(from_np(rho), retain_qubits=[0]))
        )

        rho2 = from_np(rand(4, 4) + 1j * rand(4, 4))
        assert np.allclose(
            to_np(rho2), to_np(partial_trace(rho2, retain_qubits=[1, 0]))
        )

        # lastly, test if all qubits are killed
        assert np.allclose(
            np.trace(to_np(rho2)), to_np(partial_trace(rho2, retain_qubits=[]))
        )

    def test_work_for_randommat_ndim(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        nqb = 4
        dim = 2 ** nqb
        qbs_tokeep = [1, 2]
        rho = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        rho_tmp = from_np(rho)

        def get_shouldbe1(arho):
            arho = arho.reshape([2] * nqb * 2)
            # retains [1, 2]
            # [0,1,2,3, 4,5,6,7] -> [0,1,2, 4,5,6]  (trace 3, 7)
            arho = np.trace(arho, axis1=3, axis2=7)
            # [0,1,2, 4,5,6] -> [1,2, 5,6]  (trace 0, 3)
            arho = np.trace(arho, axis1=0, axis2=3)
            return arho.reshape(4, 4)

        assert np.allclose(
            to_np(partial_trace(rho_tmp, retain_qubits=qbs_tokeep)), get_shouldbe1(rho)
        )

        qbs_tokeep = [0, 3]

        def get_shouldbe2(arho):
            arho = arho.reshape([2] * nqb * 2)
            # retains [0, 3]
            # [0,1,2,3, 4,5,6,7] -> [0,1,3, 4,5,7]  (trace 2, 6)
            arho = np.trace(arho, axis1=2, axis2=6)
            # [0,1,3, 4,5,7] -> [0,3, 4,7]  (trace 1, 4)
            arho = np.trace(arho, axis1=1, axis2=4)
            return arho.reshape(4, 4)

        assert np.allclose(
            to_np(partial_trace(rho_tmp, retain_qubits=qbs_tokeep)), get_shouldbe2(rho)
        )

    def test_nd_retain_qbs_order_does_not_matter(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        nqb = 4
        dim = 2 ** nqb
        qbs_tokeep = np.array([1, 2, 3])
        qbs_tokeep_shuffled = qbs_tokeep.copy()
        while np.array_equal(qbs_tokeep_shuffled, qbs_tokeep):
            np.random.shuffle(qbs_tokeep_shuffled)

        rho = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        rho = from_np(rho)
        assert np.allclose(
            to_np(partial_trace(rho, retain_qubits=qbs_tokeep)),
            to_np(partial_trace(rho, retain_qubits=qbs_tokeep_shuffled)),
        )

    def test_1d_and_ndim_work_the_same(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        nqb = 4
        dim = 2 ** nqb
        qb_to_keep = 2
        rho = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        rho = from_np(rho)
        assert np.allclose(
            to_np(partial_trace_1d(rho, retain_qubit=qb_to_keep)),
            to_np(partial_trace(rho, retain_qubits=(qb_to_keep,))),
        )

    def test_works_for_rand_pure_ndim(self):
        from_np = self.from_numpy
        to_np = self.to_numpy
        size = (2, 3)

        def f_rand():
            return get_random_wf(np, nqb=size[0], nwf=size[1])

        rand1, rand2, rand3 = [from_np(f_rand()) for _ in range(3)]
        rand123 = kron_each(rand1, kron_each(rand2, rand3))

        def ptwfs(iwf, retain_qubits):
            iwf = make_density_matrix(iwf)
            out = [partial_trace(iwf[i], retain_qubits) for i in range(iwf.shape[0])]
            out = [to_np(o) for o in out]
            return np.asarray(out)

        def mean_abs(array):
            return np.mean(np.abs(array))

        def assert_func(a, b):
            assert np.allclose(
                mean_abs(
                    ptwfs(rand123, retain_qubits=a) - to_np(make_density_matrix(b))
                ),
                0,
            )

        assert_func([0, 1], rand1)
        assert_func([2, 3], rand2)
        assert_func([4, 5], rand3)


@add_methods
class TestPartialTraceWf(TestCase):
    """
    Notes
    -----
    This test relies on the correctness of :class:`.TestPartialTrace`.

    """

    def setUp(self) -> None:
        self.prepare_module()

    def test_trivial_case_ndim(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        wf = get_random_wf(np, nqb=1)
        rho = np.outer(wf, wf.conj())
        o1 = partial_trace_wf(from_np(wf), retain_qubits=[0])
        assert np.allclose(rho, to_np(o1))

        wf2 = get_random_wf(np, nqb=2)
        rho2 = np.outer(wf2, wf2.conj())
        assert np.allclose(
            rho2, to_np(partial_trace_wf(from_np(wf2), retain_qubits=[1, 0]))
        )

        # lastly, test if all qubits are killed
        assert np.allclose(1, to_np(partial_trace_wf(from_np(wf2), retain_qubits=[])))

    def test_same_as_partial_trace(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        nqb = 4
        wf = get_random_wf(np, nqb=nqb)
        wf_tmp = from_np(wf)
        rho = np.outer(wf, wf.conj())

        def get_shouldbe(arho, retain):
            arho = from_np(arho)
            arho = partial_trace(arho, retain_qubits=retain)
            return to_np(arho)

        qbs_tokeep = [1, 2]
        assert np.allclose(
            to_np(partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep)),
            get_shouldbe(rho, qbs_tokeep),
        )

        qbs_tokeep = [0, 3]

        assert np.allclose(
            to_np(partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep)),
            get_shouldbe(rho, qbs_tokeep),
        )

        for qbs_tokeep in [0, 1, 2, 3]:
            qbs_tokeep = [qbs_tokeep]
            assert np.allclose(
                to_np(partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep)),
                get_shouldbe(rho, qbs_tokeep),
            )

        for qbs_tokeep in [(0, 1), (0, 1, 2), (0, 1, 2, 3)]:
            assert np.allclose(
                to_np(partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep)),
                get_shouldbe(rho, qbs_tokeep),
            )

    def test_nd_retain_qbs_order_does_not_matter(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        nqb = 4
        qbs_tokeep = np.array([1, 2, 3])
        qbs_tokeep_shuffled = qbs_tokeep.copy()
        while np.array_equal(qbs_tokeep_shuffled, qbs_tokeep):
            np.random.shuffle(qbs_tokeep_shuffled)

        wf = get_random_wf(np, nqb=nqb)
        wf = from_np(wf)

        assert np.allclose(
            to_np(partial_trace_wf(wf, retain_qubits=qbs_tokeep)),
            to_np(partial_trace_wf(wf, retain_qubits=qbs_tokeep_shuffled)),
        )


@add_methods
class TestPartialTraceWfKeepFirst(TestCase):
    """
    Notes
    ------
    This test relies on the correctness of two functions:

    - `make_density_matrix`
    - `partial_trace`

    """

    def setUp(self) -> None:
        self.prepare_module()

    def test_trivial(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        wf = get_random_wf(np, 1)
        o = partial_trace_wf_keep_first(from_np(wf), 1)
        s = np.outer(wf, wf.conj())
        # s = np.sum(wf.dot(wf.conj()))
        assert np.allclose(to_np(o), s)

    def test_keep_zero(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        wf = get_random_wf(np, 2)
        o = partial_trace_wf_keep_first(from_np(wf), 0)
        o = to_np(o)
        assert o.shape == (1, 1)
        assert np.allclose(o, [[1]])

    def test_3qb(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        wf = from_np(get_random_wf(np, 3))
        rho = make_density_matrix(wf)

        o1 = partial_trace_wf_keep_first(wf, 1)
        s1 = partial_trace(rho, retain_qubits=[0])
        assert np.allclose(to_np(o1), to_np(s1))

        o1 = partial_trace_wf_keep_first(wf, 2)
        s1 = partial_trace(rho, retain_qubits=[0, 1])
        assert np.allclose(to_np(o1), to_np(s1))


@add_methods
class TestTraceDistance(TestCase):
    @staticmethod
    def get_rand(shape=(4, 4)):
        rho = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        rho = rho + np.conjugate(np.transpose(rho))
        return rho

    @staticmethod
    def trace_distance_correct(target_rho, pred_rho):
        return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(target_rho - pred_rho)))

    def setUp(self) -> None:
        self.prepare_module()

    def test(self):
        from_np = self.from_numpy
        to_np = self.to_numpy
        get_rand = self.get_rand
        trace_distance_correct = self.trace_distance_correct

        rho1 = get_rand()
        rho2 = get_rand()
        dist_np = trace_distance_correct(rho1, rho2)
        dist_tf = to_np(trace_distance(from_np(rho1), from_np(rho2)))
        assert np.allclose(dist_np, dist_tf)

        rho1 = get_rand((2, 2))
        rho2 = get_rand((2, 2))
        dist_np = trace_distance_correct(rho1, rho2)
        dist_tf = to_np(trace_distance(from_np(rho1), from_np(rho2)))
        assert np.allclose(dist_np, dist_tf)

    def test_1qb(self):
        from_np = self.from_numpy
        to_np = self.to_numpy
        get_rand = self.get_rand
        trace_distance_correct = self.trace_distance_correct

        # test the special one just for 1 qubit system
        rho1 = get_rand((2, 2))
        rho2 = get_rand((2, 2))
        dist_np = trace_distance_correct(rho1, rho2)
        dist_tf = to_np(trace_distance_1qb(from_np(rho1), from_np(rho2)))
        assert np.allclose(dist_np, dist_tf)

    def test_with_svd(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        def get_rand(shape=(4, 4)):
            rho = np.random.rand(*shape) + 1j * np.random.rand(*shape)
            rho = rho + np.conjugate(np.transpose(rho))
            return rho

        def trace_distance_(target_rho, pred_rho):
            return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(target_rho - pred_rho)))

        rho1 = get_rand()
        rho2 = get_rand()
        dist_np = trace_distance_(rho1, rho2)
        dist_tf = to_np(trace_distance_using_svd(from_np(rho1), from_np(rho2)))
        assert np.allclose(dist_np, dist_tf)

        rho1 = get_rand((2, 2))
        rho2 = get_rand((2, 2))
        dist_np = trace_distance_(rho1, rho2)
        dist_tf = to_np(trace_distance_using_svd(from_np(rho1), from_np(rho2)))
        assert np.allclose(dist_np, dist_tf)

        # test the special one just for 1 qubit system
        rho1 = get_rand((2, 2))
        rho2 = get_rand((2, 2))
        dist_np = trace_distance_(rho1, rho2)
        dist_tf = to_np(trace_distance_using_svd(from_np(rho1), from_np(rho2)))
        assert np.allclose(dist_np, dist_tf)
