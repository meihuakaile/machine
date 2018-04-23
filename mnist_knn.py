# -*- coding:utf-8 -*-
# __author__='chenliclchen'

import scipy.io as scio
import operator

train_data_file = "./data1/train/mnist_train.mat"
train_label_file = "./data1/train/mnist_train_labels.mat"
test_data_file = "./data1/test/mnist_test.mat"
test_label_file = "./data1/test/mnist_test_labels.mat"

train_data = scio.loadmat(train_data_file)['mnist_train']
train_label = scio.loadmat(train_label_file)['mnist_train_labels']
test_data = scio.loadmat(test_data_file)['mnist_test']
test_label = scio.loadmat(test_label_file)['mnist_test_labels']


def classify(one_test_data, train_data, train_label, k):
    diff = one_test_data - train_data
    sq_diff = diff ** 2
    sq_distances = sq_diff.sum(axis=1)
    distances = sq_distances ** 0.5
    before_ind = distances.argsort()  # argsort  distances 排序之后数据在原来数据组的下标
    class_count = {}

    for ind in range(0, k):
        label = train_label[before_ind[ind]][0]
        class_count[label] = class_count.get(label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


for test_ind in range(test_label.size):
    test = 0
    truth_label = int(classify(test_data[test_ind], train_data, train_label, 3))
    predict_label = int(test_label[test_ind][0])
    if test_ind % 500 == 0:
        print "deal : ", test_ind
    if truth_label != predict_label:
        print "predict is ", predict_label, " the truth is ", truth_label, " test index is ", test_ind
