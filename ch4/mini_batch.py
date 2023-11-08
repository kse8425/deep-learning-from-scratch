from dataset.mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=False)
    print(x_train.shape)
    print(t_train.shape)


if __name__ == '__main__':
    main()
