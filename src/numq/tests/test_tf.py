import tensorflow as tf

from numq import (
    load_module,
)
from numq.tests.test_numpy import (
    TestKronEach,
    TestLoadStatesToLargerHilbertSpace,
    TestMakeDensityMatrix,
    TestOverlap,
    TestPartialTrace,
    TestTraceDistance,
    TestPartialTraceWf,
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
