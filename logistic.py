# -*- coding:utf-8 -*-
# __author__='chenliclchen'
# 为了使/有小数除、//有不大于的最大正数的效果
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from load_data2 import load_datas as load_data2
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
def grad_ascent(data, label, params, test_data, test_label):
    max_iter = params['max_iter']
    alpha = params['alpha']
    data = np.mat(data)
    one_data_length = data.shape[1]
    iter_weight = np.mat(np.ones((one_data_length, 1)))
    error_rates = []
    for ind in xrange(max_iter):
        data_iter = sigmoid(data * iter_weight)
        diff = label - data_iter
        iter_weight = iter_weight + alpha * data.transpose() * diff
        error_rate = test(test_data, test_label, iter_weight)
        error_rates.append(error_rate)
        print 'iter :', ind, "error rate: ", error_rate

    return iter_weight, error_rates

# alpha 学习率
def grad_ascent1(data, label, params, test_data, test_label):
    max_iter = params['max_iter']
    alpha = params['alpha']
    data = np.mat(data)
    one_data_length = data.shape[1]
    iter_weight = np.mat(np.ones((one_data_length, 1)))
    error_rates = []
    for ind in xrange(max_iter):
        index = range(data.shape[0])
        for i in xrange(data.shape[0]):
            alpha = 4 / (ind+i+1.0) + 0.0001
            rand_index = int(np.random.uniform(0, len(index)))
            data_iter = sigmoid(data[rand_index] * iter_weight)
            diff = label[rand_index] - data_iter
            iter_weight = iter_weight + alpha * data[rand_index].transpose() * diff
            del(index[rand_index])
        error_rate = test(test_data, test_label, iter_weight)
        error_rates.append(error_rate)
        print 'iter :', ind, "error rate: ", error_rate

    return iter_weight, error_rates

def recognition(data, weight):
    data = np.mat(data)
    weight = np.mat(weight)
    h = sigmoid(data * weight)
    if h > 0.5:
        return 1
    else:
        return 0


def test(test_data, test_label, weight_result):
    err = 0
    for ind in xrange(test_label.shape[0]):
        result = recognition(test_data[ind], weight_result)
        if test_label[ind] != result:
            # print 'label: ', test_label[ind], 'recognition:', result
            err += 1
    return err / test_label.shape[0]

def draw_loss(loss):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(100), loss, 'g')
    plt.xlabel('iter')
    plt.ylabel('loss')
    for x, y in zip(range(100), loss):
        plt.text(x, y, round(y, 2), ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    params = {"max_iter": 100, "alpha": 0.07}
    train_data, train_label, test_data, test_label = load_data2()

    ################# data1数据集有10类，需要二分。如果想把10个类全部分出来，需要进行多次二分
    # train_label = binary(train_label)
    # test_label = binary(test_label)
    # 增加b部分
    train_b = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((train_b, train_data))
    test_b = np.ones((test_data.shape[0], 1))
    test_data = np.hstack((test_b, test_data))

    from load_data3 import load_data
    train_data, train_label = load_data()
    test_data, test_label = load_data(file_path='./horseColic/horseColicTest.txt')
    weight_result, loss = grad_ascent1(train_data, train_label, params, test_data, test_label)
    draw_loss(loss)
    #
    # for ind in xrange(weight_result.shape[0]):
    #     print weight_result[ind][0]


    # data1 error rate:  0.0263
    # data2 error rate:  0.0235294117647
