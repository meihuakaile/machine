# -*- coding:utf-8 -*-
# __author__='chenliclchen'
# 为了使/有小数除、//有不大于的最大正数的效果
from __future__ import division
import numpy as np
from load_data2 import load_datas as load_data2
from load_data1 import load_datas as load_data1
from scipy.special import expit


def binary(label):
    for ind in xrange(label.shape[0]):
        if label[ind][0] != 0:
            label[ind][0] = 1
    return label


def sigmoid(z):
    # return 1.0 / (1 + np.exp(-z))
    return expit(z)


# alpha 学习率
def grad_ascent(data, label, params):
    max_iter = params['max_iter']
    alpha = params['alpha']
    data = np.mat(data)
    one_data_length = data.shape[1]
    iter_weight = np.mat(np.ones((one_data_length, 1)))
    for ind in xrange(max_iter):
        data_iter = sigmoid(data * iter_weight)
        diff = label - data_iter
        iter_weight = iter_weight + alpha * data.transpose() * diff

    return iter_weight


def recognition(data, weight):
    data = np.mat(data)
    weight = np.mat(weight)
    h = sigmoid(data * weight)
    if h > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    params = {"max_iter": 8, "alpha": 0.07}
    train_data, train_label, test_data, test_label = load_data2()

    ################# data1数据集有10类，需要二分。如果想把10个类全部分出来，需要进行多次二分
    # train_label = binary(train_label)
    # test_label = binary(test_label)
    # 增加b部分
    train_b = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((train_b, train_data))
    test_b = np.ones((test_data.shape[0], 1))
    test_data = np.hstack((test_b, test_data))
    weight_result = grad_ascent(train_data, train_label, params)

    for ind in xrange(weight_result.shape[0]):
        print weight_result[ind][0]
    err = 0
    for ind in xrange(test_label.shape[0]):
        result = recognition(test_data[ind], weight_result)
        if test_label[ind] != result:
            print 'label: ', test_label[ind], 'recognition:', result
            err += 1
    print "error rate: ", err / test_label.shape[0]
    # data1 error rate:  0.0263
    # data2 error rate:  0.0235294117647
