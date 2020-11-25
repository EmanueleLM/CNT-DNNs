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
file_ = "./weights/CIFAR10/CIFAR10_small_cnn_nlayers-4_init-random-uniform_support-0.05_seed-0_realaccuracy-0.5296_binaccuracy-0.5250.npy"
W = np.load(file_, allow_pickle=True)  # load parameters
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
for l in range(num_layers):
    link_weights = np.concatenate((CNet.weights[l].flatten(), CNet.biases[l]))
    min_, max_ = np.min(link_weights), np.max(link_weights)
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    density = stats.kde.gaussian_kde(link_weights)
    plt.plot(x, density(x), alpha=.5, color='blue')
    plt.title("{} Link Weights LAYER {}".format(dataset, l))
    plt.xlabel("W")
    plt.ylabel("PDF(W)")
    fig_name = "CIFAR10_small_cnn_link-weights_layer{}_acc0.52.png".format(l)
    plt.savefig("./results/images/CIFAR10/single_networks/" + fig_name)
    plt.show()

# Strength
file_ = "./weights/CIFAR10/CIFAR10_small_cnn_nlayers-4_init-random-uniform_support-0.05_seed-0_realaccuracy-0.5296_binaccuracy-0.5250.npy"
W = np.load(file_, allow_pickle=True)  # load parameters
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
for l in range(num_layers):
    nodes_strength = CNet.nodes_strength(l).flatten()
    nodes_fluctuation = CNet.nodes_fluctuation(l)
    min_, max_ = np.min(nodes_strength), np.max(nodes_strength)
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    density = stats.kde.gaussian_kde(nodes_strength)
    plt.plot(x, density(x), alpha=.5, color='blue')
    plt.fill_between(x, density(x)-nodes_fluctuation, density(x)+nodes_fluctuation, alpha=.1, color='blue')
    plt.title("{} Nodes Strength (std) LAYER {}".format(dataset, l))
    plt.xlabel("s")
    plt.ylabel("PDF(s)")
    fig_name = "CIFAR10_small_cnn_nodes-strength_layer{}_acc0.52.png".format(l)
    plt.savefig("./results/images/CIFAR10/single_networks/" + fig_name)
    plt.show()
    print("Nodes Fluctuation Layer {}: {}".format(l, nodes_fluctuation))