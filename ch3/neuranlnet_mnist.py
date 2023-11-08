import pickle

import numpy as np

from ch3.functions import sigmoid, softmax
from dataset.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)
    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)
    a3 = z2.dot(W3) + b3
    y = softmax(a3)

    return y


def main():
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_count = 0

    for i in range(0, len(x), batch_size):
        y = predict(network, x[i:i + batch_size])
        p = np.argmax(y, axis=1)
        accuracy_count += np.sum(p == t[i:i + batch_size])

    print('Accuracy : ', str(float(accuracy_count / len(x))))


if __name__ == '__main__':
    main()
