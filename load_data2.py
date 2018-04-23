# -*- coding:utf-8 -*-
# __author__='chenliclchen'

import os
import numpy as np

TRAIN_FILE = "./data2/train/"
TEST_FILE = "./data2/test/"


def load_data(data_path):
    data_paths = os.listdir(data_path)
    data_num = len(data_paths)
    data_array = np.zeros((data_num, 1024))
    label_array = np.zeros((data_num, 1))
    for idx in xrange(data_num):
        item = data_paths[idx]
        file_object = open(data_path + item)
        file_content = file_object.read()
        data_tmp = file_content.replace("\n", "")
        data_content = np.array(list(data_tmp))
        data_array[idx] = data_content
        item_tmp = item.split("_")
        label_array[idx][0] = item_tmp[0]
    return data_array, label_array


def load_datas(train_path=TRAIN_FILE, test_path=TEST_FILE):
    train_data, train_label = load_data(train_path)
    test_data, test_label = load_data(test_path)
    return train_data, train_label, test_data, test_label
