# -*- coding:utf-8 -*-
# __author__='chenliclchen'
from __future__ import division
import os
import pickle


# 统计每个类别每个单词出现的次数
# （该类别文档数） / （所有类别文档数）
# （该类别该单词个数 + 1）/（该类别单词个数 + 所有类别无重复单词个数）

# step1，统计每个类别每个单词出现的次数
def count_every_sort_every_word(sort_path):
    count = {}
    for item in os.listdir(sort_path):
        file_path = os.path.join(sort_path, item)
        with open(file_path) as file:
            words = file.readlines()
        for word in words:
            word = word.replace('\r\n', '')
            if word in count.keys():
                count[word] += 1
            else:
                count[word] = 1
    return count


# 保存step1的结果
def save_every_sort_every_word(count_dict, sort_path):
    file = open(sort_path, 'wb')
    pickle.dump(count_dict, file)
    file.close()


# 计算所有类别不重复单词总数， 总词数包括重复
def count_all_words(count_path):
    all_words_no_repeat = set()
    all_words_count = 0
    for item in os.listdir(count_path):
        file = open(os.path.join(count_path, item), 'rb')
        one_sort = pickle.load(file)
        all_words_no_repeat.update(one_sort.keys())
        for key in one_sort.keys():
            all_words_count += one_sort[key]

    return len(all_words_no_repeat), all_words_count


# 每个类别样本数量
def count_every_sort_docu(sort_path):
    count_docu = {}
    all_docu_num = 0
    for item in os.listdir(sort_path):
        one_sort_path = os.path.join(sort_path, item)
        count_docu[item] = len(os.listdir(one_sort_path))
        all_docu_num += count_docu[item]
    count_docu['all'] = all_docu_num
    return count_docu


# 返回某个类别所有词的数据量
def count_one_sort(sort_path):
    all_words_count = 0
    file = open(sort_path, 'rb')
    one_sort = pickle.load(file)
    file.close()
    for key in one_sort.keys():
        all_words_count += one_sort[key]
    return all_words_count


def get_test_words(test_path):
    file = open(test_path)
    words = file.readlines()
    file.close()
    for ind, word in enumerate(words):
        words[ind] = word.replace('\r\n', '')
    return words


if __name__ == '__main__':
    base_path = './NBCorpus'
    sort_path = os.path.join(base_path, 'train')
    count_path = os.path.join(base_path, 'count')
    # 计算并保存step1的数据
    for one_sort in os.listdir(sort_path):
        one_sort_path = os.path.join(sort_path, one_sort)
        one_sort_count = count_every_sort_every_word(one_sort_path)
        sort_name = os.path.split(one_sort_path)
        one_count_path = os.path.join(count_path, sort_name[-1])
        save_every_sort_every_word(one_sort_count, one_count_path)

    sort_prob = {}
    test_path = os.path.join(base_path, 'test', '487141newsML.txt')
    test_words = get_test_words(test_path)
    all_words_num, all_words_num_repeat = count_all_words(count_path)
    every_docu = count_every_sort_docu(sort_path)
    for item in os.listdir(count_path):
        one_sort_count_path = os.path.join(count_path, item)
        one_sort_num = count_one_sort(one_sort_count_path)
        file = open(one_sort_count_path, 'rb')
        one_sort_count = pickle.load(file)
        prior = every_docu[item] / every_docu['all']
        import math
        # 加入ln操作
        all_words_prob = math.log(prior)
        for word in test_words:
            word_num = one_sort_count.get(word, 0)
            word_prob = (word_num + 1) / (one_sort_num + all_words_num)
            # all_words_prob *= word_prob
            # 加入ln操作
            all_words_prob += math.log(word_prob)
        sort_prob[item] = all_words_prob
        file.close()
    print filter(lambda x:max(sort_prob.values()) == sort_prob[x], sort_prob)[0]