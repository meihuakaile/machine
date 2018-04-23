# -*- coding:utf-8 -*-
# __author__='chenliclchen'

import scipy.io as scio
import Image as image
from matplotlib import pyplot as plt

train_file = "./mnist_train.mat"
train_label = "./mnist_train_labels.mat"
train_dict = scio.loadmat(train_file)
train_label_dict = scio.loadmat(train_label)
train_data = train_dict['mnist_train']
train_label = train_label_dict['mnist_train_labels']
print train_data.shape, train_label.shape
img1_data = train_data[0].reshape(28, 28)
img1 = image.fromarray(img1_data)
# img1.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img1_data, cmap="gray")
plt.show()


