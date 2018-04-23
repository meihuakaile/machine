# -*- coding:utf-8 -*-
# __author__='chenliclchen'

import numpy as np
from scipy.special import expit, logit


def sigmoid(x):
    return expit(x)


def reverser_sigmoid(x):
    return logit(x)


class Network(object):
    def __init__(self, layers, bias=True):
        self.layers = []
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            current_layer = Layer(layer, layers[i - 1], bias)
            self.layers.append(current_layer)


class Layer(object):
    def __init__(self, layer, last_layer, bias):
        if bias:
            self.v = np.random.uniform(-0.5, 0.5, (layer, last_layer + 1))
        else:
            self.v = np.random.uniform(-0.5, 0.5, (layer, last_layer))
        self.func = (sigmoid, reverser_sigmoid)
        self.input_data = None
        self.output_data = None
