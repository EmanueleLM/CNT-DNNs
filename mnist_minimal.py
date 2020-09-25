# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:49:11 2020

@author: Emanuele
"""
from __future__ import print_function
import keras
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras import backend as K

seed_value = 42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# import data, once
batch_size = 128
num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

initializers = {}
initializers['glorot-normal'] = keras.initializers.glorot_normal(seed=seed_value)
num_kernels = 5
kernel_size = 28
model = Sequential()
model.add(Conv2D(num_kernels, kernel_size=(kernel_size, kernel_size),activation='relu', input_shape=input_shape, kernel_initializer=initializers['glorot-normal'], bias_initializer='zeros'))
model.add(Flatten())
#model.add(Dense(kernel_size, activation='relu', kernel_initializer=initializers['glorot-normal'], bias_initializer='zeros'))
model.add(Dense(32, activation='relu', kernel_initializer=initializers['glorot-normal'], bias_initializer='zeros'))
model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers['glorot-normal'], bias_initializer='zeros'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

# show the kernels
if num_kernels == 1:
    kernels = model.layers[0].get_weights()[0].reshape(kernel_size, kernel_size, 1)
else:
    kernels = model.layers[0].get_weights()[0].squeeze()
min_, max_ = kernels.min(), kernels.max()
mean, std = kernels.mean(), kernels.std()
for i in range(num_kernels):
    plt.imshow(kernels[:,:,i], cmap='gray', vmin=min_, vmax=max_)
    plt.show()
# generate same number of random, gaussian kernels
for i in range(num_kernels):
    plt.imshow(np.random.normal(mean, std, kernels[:,:,i].shape), cmap='gray', vmin=min_, vmax=max_)
    plt.show()

