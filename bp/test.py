# -*- coding:utf-8 -*-
# __author__='chenliclchen'

from mat.load_data1 import load_datas
# from ..load_data1 import load_datas
from network import Network
from train import Train


def show_test(test_data, test_label):
    right_sum = sum([1 for ind, label in enumerate(test_data) if test_label[ind] == label])
    print "right :", right_sum, "all: ", len(test_label)
    print 1.0 * right_sum / len(test_label)


def test(epochs=10, learning_rate=0.05):
    train_data, train_label, test_data, test_label = load_datas()
    network = Network([784, 250, 100, 10])
    train = Train()
    once = len(train_data) / epochs
    for ind in range(epochs):
        train.forward(network.layers, train_data[once * ind: once * (ind + 1)])
        train.backward(network.layers, train_label[once * ind: once * (ind + 1)], learning_rate)
        result = train.predict(network.layers, test_data)
        print ind, "epochs:"
        show_test(result, test_label)


if __name__ == "__main__":
    test()
