import numpy as np
from PIL import Image
from numpy import ndarray


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def identify_function(x):
    return x


def softmax(a: ndarray):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
