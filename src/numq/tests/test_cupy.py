import cupy as cp

from numq import (
    load_module,
    get_random_wf,
    make_density_matrix,
    partial_trace,
    partial_trace_wf_keep_first,
)
from numq.tests.test_numpy import TestKronEach, \
    TestLoadStatesToLargerHilbertSpace, TestMakeDensityMatrix, TestOverlap, \
    TestPartialTrace, TestPartialTraceWf, TestPartialTraceWfKeepFirst, \
    TestTraceDistance


def add_methods(cls):
    from types import MethodType

    # noinspection PyUnusedLocal
    def from_numpy(self, a):
        return cp.asarray(a)

    # noinspection PyUnusedLocal
    def to_numpy(self, a):
        return a.get()

    # noinspection PyUnusedLocal
    def prepare_module(self):
        load_module('cupy')

    cls.from_numpy = MethodType(from_numpy, cls)

    cls.to_numpy = MethodType(to_numpy, cls)

    cls.prepare_module = MethodType(prepare_module, cls)

    return cls


@add_methods
class TestKronEachCupy(TestKronEach): pass


@add_methods
class TestLoadStatesToLargerHilbertSpaceCupy(
    TestLoadStatesToLargerHilbertSpace):
    pass


@add_methods
class TestMakeDensityMatrixCupy(TestMakeDensityMatrix): pass


@add_methods
class TestOverlapCupy(TestOverlap): pass


@add_methods
class TestPartialTraceCupy(TestPartialTrace): pass


class TestPartialTraceWfCupy(TestPartialTraceWf):
    def from_numpy(self, a):
        return cp.asarray(a, dtype=cp.complex64)

    def to_numpy(self, a):
        return a.get()

    def prepare_module(self):
        load_module('cupy')


class TestPartialTraceWfKeepFirstCupy(TestPartialTraceWfKeepFirst):
    def from_numpy(self, a):
        return cp.asarray(a, dtype=cp.complex64)

    def to_numpy(self, a):
        return a.get()

    def prepare_module(self):
        load_module('cupy')

    def test_cupy_thread_dim_limit(self):
        # It is import to test when the thread dim is larger then 32 for cuda
        # code. This happens when there are more than 6 qubits to keep.
        # Note: 2 ** 5 = 32, 2 ** 6 = 64.
        # See the implementation for cuda for details.
        import numpy
        to_np = self.to_numpy

        wf = self.from_numpy(get_random_wf(numpy, 7))
        rho = make_density_matrix(wf)
        # Before boundary
        o1 = partial_trace_wf_keep_first(wf, 5)
        s1 = partial_trace(rho, retain_qubits=list(range(5)))
        assert numpy.allclose(to_np(o1), to_np(s1))

        # After boundary
        o1 = partial_trace_wf_keep_first(wf, 6)
        s1 = partial_trace(rho, retain_qubits=list(range(6)))
        assert numpy.allclose(to_np(o1), to_np(s1))


@add_methods
class TestTraceDistanceCupy(TestTraceDistance): pass
