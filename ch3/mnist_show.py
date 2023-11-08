from ch3.functions import img_show
from dataset.mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]
    img_show(img.reshape(28, 28))

    label = t_train[0]
    print(label)


if __name__ == '__main__':
    main()
