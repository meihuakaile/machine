# -*- coding:utf-8 -*-
# __author__='chenlicl.chen'

import scipy.io as scio

TRAIN_DATA_FILE = "../data1/train/mnist_train.mat"
TRAIN_LABEL_FILE = "../data1/train/mnist_train_labels.mat"
TEST_DATA_FILE = "../data1/test/mnist_test.mat"
TEST_LABEL_FILE = "../data1/test/mnist_test_labels.mat"


def load_datas():
    train_data = scio.loadmat(TRAIN_DATA_FILE)['mnist_train']
    train_label = scio.loadmat(TRAIN_LABEL_FILE)['mnist_train_labels']
    test_data = scio.loadmat(TEST_DATA_FILE)['mnist_test']
    test_label = scio.loadmat(TEST_LABEL_FILE)['mnist_test_labels']
    return train_data, train_label, test_data, test_label
