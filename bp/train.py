# -*- coding:utf-8 -*-
# __author__='chenliclchen'

import numpy as np
from network import Layer


class Train(object):
    def __init__(self):
        pass

    def forward(self, layers, input_data, bias=True):
        # 对输入执行sigmoid
        for ind, layer in enumerate(layers):
            if ind == 0:
                input_data = layer.func[0](input_data)
            if bias:
                input_data = np.hstack((input_data, np.ones((len(input_data), 1))))
            layer.input_data = input_data
            output_data = layer.func[0](np.dot(input_data, layer.v.T))
            input_data = output_data
            layer.output_data = output_data
        return output_data

    def backward(self, layers, target, learning_rate, bias=True):
        length = len(layers)
        target = layers[0].func[0](target)
        for ind, layer in enumerate(layers[::-1]):
            if ind == 0:
                diff = target - layer.output_data
            else:
                next_layer_ind = length - ind
                diff = layers[next_layer_ind].diff

            beta = np.dot((1 - layer.output_data), diff.T)  # 10 * 10
            gi = np.dot(beta.T, layer.output_data)  # 60000 * 10
            gi /= max(np.max(gi), np.min(gi) * -1)
            delta = learning_rate * np.dot(layer.input_data.T, gi)
            layer.v += delta.T
            diff_weight = np.dot(diff, layer.v)
            layer.diff = diff_weight if not bias else diff_weight[:, :-1]
            # gi = layers[next_layer_ind].gi
            # weight = layers[next_layer_ind].v
            # next_diff = np.dot(weight, gi)
            # next_sum = np.sum(next_diff, axis=1)
            # beta = np.dot(layer.output_data, (1-layer.output_data).T)
            # delta = np.dot(beta, next_sum)
            # layer

    def predict(self, layers, test_data, bias=True):
        output = self.forward(layers, test_data, bias)
        return np.argmax(output, axis=1)
