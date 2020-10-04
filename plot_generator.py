# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:52:24 2020

@author: Emanuele

Use this code to scan the results/fc (or other architectures) folder to extract metrics from all the
 raw weights, i.e., not averaged, and plot histograms of
- link weights
- node strength (input-output)
- nodes disparity/fluctuation

"""
import copy as cp
import glob
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from colour import Color

from ComplexNetwork import ComplexNetwork

# custom seed's range (multiple experiments)
parser = ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", default='fc', type=str,
                    help="Architecture (fc or cnn so far).")
parser.add_argument("-d", "--dataset", dest="dataset", default='MNIST', type=str,
                    help="Dataset prefix used to save weights (MNIST, CIFAR10).")
parser.add_argument("-l", "--layers", dest="num_layers", default=5, type=int,
                    help="Number of layers of the models considered.")
parser.add_argument("-b", "--bins", dest="bins_size", default=0.025, type=float,
                    help="Accuracy range per-bin.") 
parser.add_argument("-i", "--init", dest="init_method", default='', type=str,
                    help="Initialization method(s) considered (if left empty, all are considered).")
parser.add_argument("-scale", "--scale", dest="scale", default=0.05, type=float,
                    help="Scaling factor used to initialize weights (e.g., support of uniform distribution, std of gaussian etc.).")
parser.add_argument("-maxfiles", "--maxfiles", dest="maxfiles", default=250, type=int,
                    help="Maximum number of files considered for each bin.")

args = parser.parse_args()
architecture = args.architecture
dataset = args.dataset
num_layers = args.num_layers
bins_size = args.bins_size
scaling_factor = args.scale
init = args.init_method
maxfiles = args.maxfiles

ranges_accuracy = np.arange(0., 1.0, bins_size)
input_size, output_size = (28*28 if dataset=='MNIST' else 32*32*3), 10
init = ('*' if len(init)==0 else init)
files_pattern = "./weights/{}/{}_{}_*init-{}_support-{}*".format(dataset, dataset, architecture, init, scaling_factor)  # wildcards for architecture and accuracy
saved_images_path = "./results/images/{}/".format(dataset)
img_format = '.png'

# Set colors for plotting (green to red, low to high accuracy)
num_nets = len(glob.glob(files_pattern))
num_colors = len(ranges_accuracy)
red = Color("green")
colors = list(red.range_to(Color("red"), num_colors))

# Link weights
layers_link_weights = {}
link_weights_single_layer = {k:v for (k,v) in zip(['{:4.4f}'.format(r) for r in ranges_accuracy], [np.array([]) for _ in range(num_colors)])}
link_weights = [cp.copy(link_weights_single_layer) for _ in range(num_layers)]  # one dictionary per layer
print("\n[logger]: Generating weights histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    acc_prefix = "{:4.4f}".format(acc)
    files_ = files_pattern + 'binaccuracy-{}'.format(acc_prefix) + '*.npy'
    n_files = len(glob.glob(files_))
    print("[logger]: Collecting parameters for {} nets with accuracy {}, with wildcard {}".format(n_files, acc_prefix, files_))
    for file_ in glob.glob(files_)[:maxfiles]:
        W = np.load(file_, allow_pickle=True)  # load parameters
        CNet = ComplexNetwork(architecture, num_layers, W, input_size, output_size, flatten=True)  # simplify the weights/biases usage
        for l in range(num_layers):
            link_weights[l][acc_prefix] = np.concatenate((link_weights[l][acc_prefix], CNet.weights[l], CNet.biases[l]))
for l in range(num_layers):
    print("[logger]: Generating plot for layer {}".format(l))
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(link_weights[l][acc_prefix]) != 0:
            # Generate PDF
            min_, max_ = np.min(link_weights[l][acc_prefix]), np.max(link_weights[l][acc_prefix])
            x = np.arange(min_, max_, .001)
            density = stats.kde.gaussian_kde(link_weights[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
    plt.title("{} Link Weights LAYER {}".format(dataset, l))
    plt.xlabel("W")
    plt.ylabel("PDF(W)")
    fig_name = "{}_{}_link-weights_init-{}_support-{}_layer-{}{}".format(dataset, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
    plt.savefig(saved_images_path + fig_name)
    plt.show()
    plt.close()

# Nodes strength
link_nodes_strength_single_layer = {k:v for (k,v) in zip(['{:4.4f}'.format(r) for r in ranges_accuracy], [np.array([]) for _ in range(num_colors)])}
nodes_strength = [cp.copy(link_nodes_strength_single_layer) for _ in range(num_layers)]  # one dictionary per layer
print("\n[logger]: Generating strengths histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    acc_prefix = "{:4.4f}".format(acc)
    files_ = files_pattern + 'binaccuracy-{}'.format(acc_prefix) + '*.npy'
    n_files = len(glob.glob(files_))
    print("[logger]: Collecting parameters for {} nets with accuracy {}, with wildcard {}".format(n_files, acc_prefix, files_))
    for file_ in glob.glob(files_)[:maxfiles]:
        W = np.load(file_, allow_pickle=True)  # load parameters
        CNet = ComplexNetwork(architecture, num_layers, W, input_size, output_size, flatten=False)  # simplify the weights/biases usage
        for l in range(num_layers):
            nodes_strength[l][acc_prefix] = np.concatenate((nodes_strength[l][acc_prefix], CNet.nodes_strength(l)))
for l in range(num_layers):
    print("[logger]: Generating plot for layer {}".format(l))
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(nodes_strength[l][acc_prefix]) != 0:
            # Generate PDF
            min_, max_ = np.min(nodes_strength[l][acc_prefix]), np.max(nodes_strength[l][acc_prefix])
            x = np.arange(min_, max_, .001)
            density = stats.kde.gaussian_kde(nodes_strength[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
    plt.title("{} Nodes Strength LAYER {}".format(dataset, l))
    plt.xlabel("S")
    plt.ylabel("PDF(S)")
    fig_name = "{}_{}_nodes-strength_init-{}_support-{}_layer-{}{}".format(dataset, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
    plt.savefig(saved_images_path + fig_name)
    plt.show()
    plt.close()
    
# Nodes fluctuation
link_nodes_fluctuation_single_layer = {k:v for (k,v) in zip(['{:4.4f}'.format(r) for r in ranges_accuracy], [np.array([]) for _ in range(num_colors)])}
nodes_fluctuation= [cp.copy(link_nodes_fluctuation_single_layer) for _ in range(num_layers)]  # one dictionary per layer
print("\n[logger]: Generating fluctuations histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    print("[logger]: Collecting files...")
    acc_prefix = "{:4.4f}".format(acc)
    files_ = files_pattern + 'binaccuracy-{}'.format(acc_prefix) + '*.npy'
    n_files = len(glob.glob(files_))
    print("[logger]: Collecting parameters for {} nets with accuracy {}, with wildcard {}".format(n_files, acc_prefix, files_))
    for file_ in glob.glob(files_)[:maxfiles]:
        W = np.load(file_, allow_pickle=True)  # load parameters
        CNet = ComplexNetwork(architecture, num_layers, W, input_size, output_size, flatten=False)  # simplify the weights/biases usage
        for l in range(num_layers):
            nodes_fluctuation[l][acc_prefix] = np.concatenate((nodes_fluctuation[l][acc_prefix], CNet.nodes_fluctuation(l)))
for l in range(num_layers):
    print("[logger]: Generating plot for layer {}".format(l))
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(nodes_fluctuation[l][acc_prefix]) > 1:  # at least two elements are needed for density estimation
            # Generate PDF
            min_, max_ = np.min(nodes_fluctuation[l][acc_prefix]), np.max(nodes_fluctuation[l][acc_prefix])
            x = np.arange(min_, max_, .001)
            density = stats.kde.gaussian_kde(nodes_fluctuation[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
    plt.title("{} Nodes Fluctuation LAYER {}".format(dataset, l))
    plt.xlabel("Yi")
    plt.ylabel("PDF(Yi)")
    fig_name = "{}_{}_nodes-fluctuation_init-{}_support-{}_layer-{}{}".format(dataset, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
    plt.savefig(saved_images_path + fig_name)
    plt.show()
    plt.close()