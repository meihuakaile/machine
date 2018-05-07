# -*- coding:utf-8 -*-
# __author__='chenliclchen'

from __future__ import division
import operator
from matplotlib import pyplot as plt

# 分类
# one_test_data 要分类的数据集；train_data训练集数据；k 前k个里次数最多的类别
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

# 归一化
def auto_norm(dataset):
    max = dataset.max(axis=0)
    min = dataset.min(axis=0)
    diff = max - min
    new_dataset = (dataset - min) / diff
    # 还要对测试数据归一化，因此还要返回min、diff
    return new_dataset, min, diff

# 展示数据集的分布
def show_data(data, train_label):
    for ind, item in enumerate(train_label):
        if item == 'largeDoses':
            plt.scatter(data[ind, 0], data[ind, 1], s=20, c='r')
        elif item == 'smallDoses':
            plt.scatter(data[ind, 0], data[ind, 1], s=30, c='g')
        elif item == 'didntLike':
            plt.scatter(data[ind, 0], data[ind, 1], s=50, c='b')
    plt.show()

# 加载手绘数字mnist数据集
# from load_data1 import load_datas
# train_data, train_label, test_data, test_label = load_datas()
# 加载约会数据
from load_data3 import load_dating_data
train_data, train_label, test_data, test_label = load_dating_data()
#  norm归一化
train_data, min, diff = auto_norm(train_data)
test_data = (test_data - min) / diff
# 展示训练集的数据分布
show_data(train_data, train_label)
# 计算错误率
err_num = 0
for test_ind in range(test_label.size):
    predict_label = classify(test_data[test_ind], train_data, train_label, 3)
    truth_label = test_label[test_ind][0]
    if test_ind % 800 == 0:
        print "now index : ", test_ind
    if truth_label != predict_label:
        err_num += 1
print 'err_num: ', err_num, "error rate:", err_num / test_label.size
# 因此测试集的选择是随机的，错误率一直在变化
# error rate: 0.195 err_num:  39 norm before
# error rate: 0.065 err_num:  13 norm after