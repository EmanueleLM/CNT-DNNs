# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:01:11 2020

@author: Emanuele
"""

from __future__ import print_function
import glob
import os
import keras
import random
from argparse import ArgumentParser
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras import backend as K

# custom seed's range (multiple experiments)
parser = ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", default='cnn', type=str,
                    help="Architecture (fc or cnn so far).")
parser.add_argument("-c", "--cut-train", dest="cut_train", default=1.0, type=float,
                    help="Max ratio of the dataset randomly used at each stage (must be different from 0.).")
parser.add_argument("-d", "--dataset", dest="dataset", default='MNIST', type=str,
                    help="Dataset prefix used to save weights (MNIST, CIFAR10).")
parser.add_argument("-s", "--seed", dest="seed_range", default=0, type=int,
                    help="Seed range (from n to n+<NUM_EXPERIMENTS>).") 
parser.add_argument("-b", "--bins", dest="bins_size", default=0.025, type=float,
                    help="Accuracy range per-bin.") 
parser.add_argument("-scale", "--scale", dest="scale", default=0.05, type=float,
                    help="Scaling factor used to initialize weights (e.g., support of uniform distribution, std of gaussian etc.).")
parser.add_argument("-sims", "--sims", dest="sims", default=5000, type=int,
                    help="number of simulations executed.")
parser.add_argument("-min", "--min", dest="min", default=0.0, type=float,
                    help="Min accuracy values for final models (discard anything below).")
parser.add_argument("-max", "--max", dest="max", default=1.0, type=float,
                    help="Max accuracy values for final models (discard anything above).")

args = parser.parse_args()
architecture = args.architecture
cut_train = args.cut_train
dataset = args.dataset
seed_range = args.seed_range
bins_size = args.bins_size
scaling_factor = args.scale
sims = args.sims
min_range_fin, max_range_fin = args.min, args.max

# import data
batch_size = 2048
num_classes = 10
# input image dimensions
img_rows, img_cols = ((28, 28) if dataset=='MNIST' else (32, 32))
num_channels = (1 if dataset=='MNIST' else 3)
# the data, split between train and test sets
if dataset == 'MNIST':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset == 'CIFAR10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
    raise Exception("Dataset {} not implemented (use MNIST or CIFAR10)".format(dataset))
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], num_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], num_channels, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)
    input_shape = (img_rows, img_cols, num_channels)
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

# Set unique seed value
for seed_value in range(seed_range, seed_range+sims):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
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
    
    # parameters initializers
    initializers = {}
    initializers['random-normal'] = keras.initializers.RandomNormal(mean=0.0, stddev=scaling_factor, seed=seed_value)
    initializers['random-uniform'] = keras.initializers.RandomUniform(minval=-scaling_factor, maxval=scaling_factor, seed=seed_value)
    initializers['truncated-normal'] = keras.initializers.TruncatedNormal(mean=0.0, stddev=scaling_factor, seed=seed_value)
    initializers['variance-scaling-normal-fanin'] = keras.initializers.VarianceScaling(scale=scaling_factor, mode='fan_in', distribution='normal', seed=seed_value)
    initializers['variance-scaling-normal-fanout'] = keras.initializers.VarianceScaling(scale=scaling_factor, mode='fan_out', distribution='normal', seed=seed_value)
    initializers['variance-scaling-normal-fanavg'] = keras.initializers.VarianceScaling(scale=scaling_factor, mode='fan_avg', distribution='normal', seed=seed_value)
    initializers['variance-scaling-uniform-fanin'] = keras.initializers.VarianceScaling(scale=scaling_factor, mode='fan_in', distribution='uniform', seed=seed_value)
    initializers['variance-scaling-uniform-fanout'] = keras.initializers.VarianceScaling(scale=scaling_factor, mode='fan_out', distribution='uniform', seed=seed_value)
    initializers['variance-scaling-uniform-fanavg'] = keras.initializers.VarianceScaling(scale=scaling_factor, mode='fan_avg', distribution='uniform', seed=seed_value)

    # set initializer
    optimizers = {}
    optimizers['SGD'] = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    optimizers['adam'] = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    opt = optimizers[np.random.choice(list(optimizers.keys()))]  
    
    # set training iterations
    epochs = random.randint(1, 10)
    n_layers = 3
    
    for key in initializers.keys():
        model = Sequential()
        if architecture == 'cnn':
            for _ in range(n_layers):
                model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', kernel_initializer=initializers[key], bias_initializer=initializers[key]))
            model.add(Flatten())
        elif architecture == 'fc':
            model.add(Flatten())
            for _ in range(n_layers):
                model.add(Dense(256, activation='relu',kernel_initializer=initializers[key], bias_initializer=initializers[key]))
        elif architecture == 'rnn':
            raise NotImplementedError("{} has not been implemented yet.".format(architecture))
        elif architecture == 'attention':
            raise NotImplementedError("{} has not been implemented yet.".format(architecture))
        else:
            raise NotImplementedError("{} has not been implemented.".format(architecture))
        # Add the final layers: same for every architecture so we can analyse them together ;)
        model.add(Dense(32, activation='relu', kernel_initializer=initializers[key], bias_initializer=initializers[key]))
        model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers[key], bias_initializer=initializers[key]))
               
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
        
        # Save the weights at the first and last iteration
        dst = './weights/{}/'.format(dataset)
        dataset_size = int(cut_train*len(x_train))

        # train        
        print("[logger]: Training on {}/{} datapoints.".format(dataset_size, len(x_train)))
        model.fit(x_train[:dataset_size], y_train[:dataset_size],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        
        # test and save
        print("[CUSTOM-LOGGER]: Saving final params to file at relative path {}.".format(dst))                  
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        ranges_accuracy = np.arange(min_range_fin, max_range_fin, bins_size)        
        for r in ranges_accuracy:
            if r <= accuracy <= r + bins_size:
                acc_prefix, acc_real = "{:4.4f}".format(r), "{:4.4f}".format(accuracy)
                wildcard = "{}_{}_nlayers-{}_init-{}_support-{}_*binaccuracy-{}.npy".format(dataset, architecture, n_layers+2, key, scaling_factor, acc_prefix)
                if len(glob.glob(dst+wildcard)) <= 250:
                    net_name = "{}_{}_nlayers-{}_init-{}_support-{}_seed-{}_realaccuracy-{}_binaccuracy-{}".format(dataset, architecture, n_layers+2, key, scaling_factor, seed_value, acc_real, acc_prefix)
                    np.save(dst + net_name, np.asarray(model.get_weights()))
                break
