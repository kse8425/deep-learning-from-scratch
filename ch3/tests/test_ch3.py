import numpy as np
from numpy.testing import assert_allclose

from ch3.functions import softmax
from ch3.signal_transmission import init_network, forward


def test_signal_transmission():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    assert_allclose(y, [0.31682708, 0.69627909])


def test_softmax():
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    assert np.sum(y) == 1
