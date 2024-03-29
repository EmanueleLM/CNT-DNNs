import copy as cp
import glob
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from random import shuffle

from ComplexNetwork import ComplexNetwork
from divergence import shannon_divergence

dataset = "CIFAR10"
architecture = 'cnn'
input_shape = ((28,28,1) if dataset=="MNIST" else (32,32,3))
output_size = 10

file_ = "./weights/{}/{}_medium_cnn_nlayers-6_init-random-normal_support-0.05_seed-42_realaccuracy-0.7005_binaccuracy-0.7000.npy".format(dataset, dataset)
num_layers = 6
num_conv_layers = 3
paddings, strides = tuple([0 for _ in range(num_conv_layers)]), tuple([1 for _ in range(num_conv_layers)])

saved_accuracy = '0.70'
saved_model_size = 'small'
saved_imag_path = "./results/images/{}/".format(dataset)
img_format = '.png'

# Link weights
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
    fig_name = "{}_{}_{}_link-weights_layer{}_acc{}.png".format(dataset, saved_model_size, architecture, l, saved_accuracy)
    plt.savefig("./results/images/{}/single_networks/".format(dataset) + fig_name)
    plt.show()

# Nodes Strength and Fluctuation
W = np.load(file_, allow_pickle=True)  # load parameters
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
for l in list(range(num_layers+1)):  # Need to add 1 to take care of the output layer (!)
    nodes_strength = CNet.nodes_strength(l).flatten()
    nodes_fluctuation = CNet.nodes_fluctuation(l)
    min_, max_ = np.min(nodes_strength), np.max(nodes_strength)
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    density = stats.kde.gaussian_kde(nodes_strength)
    plt.plot(x, density(x), alpha=.5, color='blue')
    #plt.fill_between(x, density(x)-nodes_fluctuation, density(x)+nodes_fluctuation, alpha=.1, color='blue')
    plt.title("{} Nodes Strength LAYER {}".format(dataset, l))
    plt.xlabel("s")
    plt.ylabel("PDF(s)")
    fig_name = "{}_{}_{}_nodes-strength_layer{}_acc{}.png".format(dataset, saved_model_size, architecture, l, saved_accuracy)
    plt.savefig("./results/images/{}/single_networks/".format(dataset) + fig_name)
    plt.show()
    print("Nodes Fluctuation Layer {}: {}".format(l, nodes_fluctuation))

# Weights Divergence: Shannon + Squared Heatmap
W = np.load(file_, allow_pickle=True)  # load parameters
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
params = []
for l in range(num_layers):
    print("[logger]: Collecting parameters for layer {}".format(l))
    params += [np.concatenate((CNet.weights[l].flatten(), CNet.biases[l]))]
divergences = np.zeros(shape=(len(params), len(params)))
for i in range(len(params)):
    for j in range(len(params)):
        if j>i:
            break
        divergences[i,j] = divergences[j,i] = shannon_divergence(params[i], params[j])
plt.title("{} Shannon Divergence for the Link Weights(layer by layer)".format(dataset))
plt.imshow(divergences, cmap='hot', interpolation='nearest')
fig_name = "{}_{}_{}_heatmap_shannon-divergence_link-weights_acc{}.png".format(dataset, saved_model_size, architecture, saved_accuracy)
plt.savefig("./results/images/{}/single_networks/".format(dataset) + fig_name)
plt.show()
print("Shannon divergence (layer by layer) (Matrix)\n ", divergences)

# Strengths Divergence: Shannon + Squared Heatmap
W = np.load(file_, allow_pickle=True)  # load parameters
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)
params = []
for l in range(num_layers):
    print("[logger]: Collecting parameters for layer {}".format(l))
    params += [CNet.nodes_strength(l)]
divergences = np.zeros(shape=(len(params), len(params)))
for i in range(len(params)):
    for j in range(len(params)):
        if j>i:
            break
        divergences[i,j] = divergences[j,i] = shannon_divergence(params[i], params[j])
plt.title("{} Shannon Divergence for the Nodes Strength (layer by layer)".format(dataset))
plt.imshow(divergences, cmap='hot', interpolation='nearest')
fig_name = "{}_{}_{}_heatmap_shannon-divergence_nodes-strength_acc{}.png".format(dataset, saved_model_size, architecture, saved_accuracy)
plt.savefig("./results/images/{}/single_networks/".format(dataset) + fig_name)
plt.show()
print("Shannon divergence (layer by layer) (Matrix)\n ", divergences)
