import copy as cp
import glob
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from random import shuffle

from ComplexNetwork import ComplexNetwork

dataset = "CIFAR10"
architecture = 'cnn'
input_shape = (32,32,3)
output_size = 10

file_path = "./weights/{}/".format(dataset)
num_layers = 4
num_conv_layers = 2
paddings, strides = (0,0), (1,1)

saved_imag_path = "./results/images/{}/".format(dataset)
img_format = '.png'

# Link weights
file_ = "./weights/CIFAR10/CIFAR10_small_cnn_nlayers-4_init-random-normal_support-0.05_seed-0_realaccuracy-0.4726_binaccuracy-0.4500.npy"
W = np.load(file_, allow_pickle=True)  # load parameters
link_weights = [np.array([]) for _ in range(num_layers)]
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
for l in range(num_layers):
    link_weights[l] = np.concatenate((link_weights[l], CNet.weights[l].flatten(), CNet.biases[l].flatten()))
    min_, max_ = np.min(link_weights[l]), np.max(link_weights[l])
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    density = stats.kde.gaussian_kde(link_weights[l])
    plt.plot(x, density(x), alpha=.5, color='blue')
    plt.title("{} Link Weights LAYER {}".format(dataset, l))
    plt.xlabel("W")
    plt.ylabel("PDF(W)")
    plt.show()

# Strength
file_ = "./weights/CIFAR10/CIFAR10_small_cnn_nlayers-4_init-random-normal_support-0.05_seed-0_realaccuracy-0.4726_binaccuracy-0.4500.npy"
W = np.load(file_, allow_pickle=True)  # load parameters
nodes_strength = [np.array([]) for _ in range(num_layers)]
nodes_fluctuation = [np.array([]) for _ in range(num_layers)]
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
for l in range(num_layers):
    nodes_strength[l] = np.concatenate((nodes_strength[l], CNet.nodes_strength(l)))
    nodes_fluctuation[l] = CNet.nodes_fluctuation(l)
    min_, max_ = np.min(nodes_strength[l]), np.max(nodes_strength[l])
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    density = stats.kde.gaussian_kde(nodes_strength[l])
    plt.plot(x, density(x), alpha=.5, color='blue')
    #plt.fill_between(x, density(x)-nodes_fluctuation[l], density(x)+nodes_fluctuation[l], alpha=.1)
    plt.title("{} Nodes Strength LAYER {}".format(dataset, l))
    plt.xlabel("s")
    plt.ylabel("PDF(s)")
    plt.show()
    print("Nodes Fluctuation Layer {}: {}".format(l, nodes_fluctuation[l]))
