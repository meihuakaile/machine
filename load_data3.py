# -*- coding:utf-8 -*-
# __author__='chenliclchen'
import numpy as np

# 加载数据，默认返回horseColic的训练数据集。数据集文件内容必须是 特征1 特征2 .... 类别
# file_path是数据集存储文件的地址；label_type是标签的数据类型，默认是浮点型；append是数据集是否需要扩展一列，比如逻辑回归需要
def load_data(file_path='./horseColic/horseColicTraining.txt', label_type=float, append=True):
    file = open(file_path)
    data = []
    label = []
    for line in file.readlines():
        one_data_list = line.strip().split()
        one_data = one_data_list[:-1]
        if append:
            one_data.append('1.0')
        data.append(one_data)
        label.append(one_data_list[-1])

    return np.array(data, dtype=float), np.array(label, dtype=label_type).reshape((len(label), 1))

# 加载数据，并分成训练集/测试集
# test_data_ratio是测试集数据量占总数据集的比例，默认0.2； file_path是加载数据的地址，默认是约会数据；label_type是label的数据类型
def load_dating_data(test_data_ratio=0.2, file_path='./dating/datingTestSet.txt', label_type=np.string_):
    all_data, all_label = load_data(file_path, label_type, append=False)
    test_num = int(test_data_ratio * all_data.shape[0])
    index = range(all_data.shape[0])
    import random
    random.shuffle(index)
    return all_data[index[test_num:], :], all_label[index[test_num:], :], all_data[index[:test_num], :], all_label[index[:test_num], :]

# data, label = load_dating_data(test_data_ratio=0.2, file_path='./dating/datingTestSet.txt', data_type=np.string_)


# data, label = load_data(file_path='./dating/datingTestSet.txt', data_type=np.string_)
