from unittest import TestCase

import numpy as np
import tensorflow as tf

from numq import (
    load_module,
    partial_trace,
)
from numq.tests.test_numpy import (
    TestKronEach,
    TestLoadStatesToLargerHilbertSpace,
    TestMakeDensityMatrix,
    TestOverlap,
    TestPartialTrace,
    TestTraceDistance,
    TestPartialTraceWf,
    get_random_wf,
)
from numq.tf_impl import (
    compute_trace_out_indices,
    partial_trace_wf_tf_autographable,
)

try:
    # tf v1 code
    global_sess = tf.InteractiveSession()
except AttributeError:
    global_sess = None


def add_methods(cls):
    from types import MethodType

    cls.sess = global_sess

    # noinspection PyUnusedLocal
    def from_numpy(self, a):
        return tf.convert_to_tensor(a)

    # noinspection PyUnusedLocal
    def to_numpy(self, a):
        if cls.sess is None:
            # tf v2
            return a.numpy()

        # tf v1
        with cls.sess.as_default():
            return a.eval()

    # noinspection PyUnusedLocal
    def prepare_module(self):
        load_module("tf")

    cls.from_numpy = MethodType(from_numpy, cls)

    cls.to_numpy = MethodType(to_numpy, cls)

    cls.prepare_module = MethodType(prepare_module, cls)

    return cls


@add_methods
class TestKronEachTF(TestKronEach):
    pass


@add_methods
class TestLoadStatesToLargerHilbertSpaceTF(TestLoadStatesToLargerHilbertSpace):
    pass


@add_methods
class TestMakeDensityMatrixTF(TestMakeDensityMatrix):
    pass


@add_methods
class TestOverlapTF(TestOverlap):
    pass


@add_methods
class TestPartialTraceTF(TestPartialTrace):
    def test_1d_and_ndim_work_the_same(self):
        pass  # TODO: tensorflow does not implement this.


@add_methods
class TestTraceDistanceTF(TestTraceDistance):
    pass


@add_methods
class TestPartialTraceWfTF(TestPartialTraceWf):
    pass


class TestPartialTraceWfAutoGraph(TestCase):
    """
    Notes
    -----
    This test relies on the correctness of :class:`.TestPartialTrace`.

    """

    def setUp(self) -> None:
        load_module("numpy")
        load_module("tf")

    def from_numpy(self, a):
        return tf.convert_to_tensor(a)

    def to_numpy(self, a):
        if global_sess is None:
            return a.numpy()
        else:
            with global_sess.as_default():
                return a.eval()

    def partial_trace_wf(self, wf, retain_qubits, nqb):
        return partial_trace_wf_tf_autographable(
            iwf=wf, nqb=nqb, retain_qubits=tuple(retain_qubits), trace_out_indices=compute_trace_out_indices(
                nqb, retain_qubits)
        )

    def test_trivial_case_ndim(self):
        from_np = self.from_numpy
        to_np = self.to_numpy

        wf = get_random_wf(np, nqb=1)
        rho = np.outer(wf, wf.conj())
        o1 = self.partial_trace_wf(from_np(wf), retain_qubits=[0], nqb=1)
        assert np.allclose(rho, to_np(o1))

        wf2 = get_random_wf(np, nqb=2)
        rho2 = np.outer(wf2, wf2.conj())
        assert np.allclose(
            rho2, to_np(self.partial_trace_wf(from_np(wf2), retain_qubits=[1, 0], nqb=2))
        )

        # lastly, test if all qubits are killed
        assert np.allclose(1, to_np(self.partial_trace_wf(from_np(wf2), retain_qubits=[], nqb=2)))

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
            to_np(self.partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep, nqb=nqb)),
            get_shouldbe(rho, qbs_tokeep),
        )

        qbs_tokeep = [0, 3]

        assert np.allclose(
            to_np(self.partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep, nqb=nqb)),
            get_shouldbe(rho, qbs_tokeep),
        )

        for qbs_tokeep in [0, 1, 2, 3]:
            qbs_tokeep = [qbs_tokeep]
            assert np.allclose(
                to_np(self.partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep, nqb=nqb)),
                get_shouldbe(rho, qbs_tokeep),
            )

        for qbs_tokeep in [(0, 1), (0, 1, 2), (0, 1, 2, 3)]:
            assert np.allclose(
                to_np(self.partial_trace_wf(wf_tmp, retain_qubits=qbs_tokeep, nqb=nqb)),
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
            to_np(self.partial_trace_wf(wf, retain_qubits=qbs_tokeep, nqb=nqb)),
            to_np(self.partial_trace_wf(wf, retain_qubits=qbs_tokeep_shuffled, nqb=nqb)),
        )
