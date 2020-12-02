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
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from colour import Color
from random import shuffle

from ComplexNetwork import ComplexNetwork
from divergence import shannon_divergence

# disable plot
matplotlib.use('Agg')  

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
parser.add_argument("-maxfiles", "--maxfiles", dest="maxfiles", default=500, type=int,
                    help="Maximum number of files considered for each bin.")
parser.add_argument("-netsize", "--netsize", dest="netsize", default='small', type=str,
                    help="Number of parameters in the hidden layers (depends on the architecture to have models with different magnitude of parameters): values are 'small', 'medium' and 'large'")

args = parser.parse_args()
architecture = args.architecture
dataset = args.dataset
num_layers = args.num_layers
bins_size = args.bins_size
scaling_factor = args.scale
init = args.init_method
maxfiles = args.maxfiles
netsize = args.netsize

num_weights = num_layers-1  # a two layers nn has one set of parameters ;)
ranges_accuracy = np.arange(0., 1.0, bins_size)
input_size, output_size = (28*28 if dataset=='MNIST' else 32*32*3), 10
init = ('*' if len(init)==0 else init)
files_pattern = "./weights/{}/{}_{}_{}_*init-{}_support-{}*".format(dataset, dataset, netsize, architecture, init, scaling_factor)  # wildcards for architecture and accuracy
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
link_weights = [cp.copy(link_weights_single_layer) for _ in range(num_weights)]  # one dictionary per layer
link_weights_densities = []  # Collect densities (to estimate divergence between )
print("\n[logger]: Generating weights histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    acc_prefix = "{:4.4f}".format(acc)
    extended_acc_prefix = ["{:4.4f}".format(a) for a in np.arange(acc, acc+bins_size, 0.025)]
    files_ = [files_pattern + 'binaccuracy-{}'.format(ap) + '*.npy' for ap in extended_acc_prefix]
    n_files = sum([len(glob.glob(f)) for f in files_])
    print("[logger]: Collecting parameters for {} nets with accuracy {}, with wildcard {}".format(n_files, acc_prefix, files_))
    processed_files, idx_glob, global_files = 0, 0, list(itertools.chain.from_iterable([glob.glob(f) for f in files_]))
    if len(global_files) > 0:  # random shuffle if non-empty
        shuffle(global_files)
    while processed_files != len(global_files):
        processed_files += 1
        file_ = global_files[idx_glob]
        W = np.load(file_, allow_pickle=True)  # load parameters
        if  np.any([np.isnan(w).any() for w in W]):
            continue
        CNet = ComplexNetwork(architecture, num_weights, 0, W, input_size, output_size, strides=None, paddings=None, flatten=True)  # simplify the weights/biases usage
        for l in range(num_weights):
            link_weights[l][acc_prefix] = np.concatenate((link_weights[l][acc_prefix], CNet.weights[l], CNet.biases[l]))
        if processed_files >= maxfiles:
            break
        idx_glob += 1
for l in range(num_weights):
    print("[logger]: Generating plot for layer {}".format(l))
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(link_weights[l][acc_prefix]) != 0:
            # Generate PDF
            min_, max_ = np.min(link_weights[l][acc_prefix]), np.max(link_weights[l][acc_prefix])
            x = np.arange(min_, max_, abs(max_-min_)/1000)
            density = stats.kde.gaussian_kde(link_weights[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
            # Collect densities
            link_weights_densities += [(x, density)]  # append data and density *function*
    plt.title("{} Link Weights LAYER {}".format(dataset, l))
    plt.xlabel("W")
    plt.ylabel("PDF(W)")
    fig_name = "{}_{}_{}_link-weights_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
    plt.savefig(saved_images_path + fig_name)
    #plt.show()
    plt.close()

# Nodes strength
nodes_strength_single_layer = {k:v for (k,v) in zip(['{:4.4f}'.format(r) for r in ranges_accuracy], [np.array([]) for _ in range(num_colors)])}
nodes_strength = [cp.copy(nodes_strength_single_layer) for _ in range(num_layers)]  # one dictionary per layer
nodes_strength_densities = []  # Collect densities (to estimate divergence between )
print("\n[logger]: Generating strengths histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    acc_prefix = "{:4.4f}".format(acc)
    extended_acc_prefix = ["{:4.4f}".format(a) for a in np.arange(acc, acc+bins_size, 0.025)]
    files_ = [files_pattern + 'binaccuracy-{}'.format(ap) + '*.npy' for ap in extended_acc_prefix]
    n_files = sum([len(glob.glob(f)) for f in files_])
    print("[logger]: Collecting parameters for {} nets with accuracy {}, with wildcard {}".format(n_files, acc_prefix, files_))
    processed_files, idx_glob, global_files = 0, 0, list(itertools.chain.from_iterable([glob.glob(f) for f in files_]))
    if len(global_files) > 0:  # random shuffle if non-empty
        shuffle(global_files)
    while processed_files != len(global_files):
        processed_files += 1
        file_ = global_files[idx_glob]
        W = np.load(file_, allow_pickle=True)  # load parameters
        if  np.any([np.isnan(w).any() for w in W]):
            continue
        CNet = ComplexNetwork(architecture, num_layers, 0, W, input_size, output_size, strides=None, paddings=None, flatten=False)  # simplify the weights/biases usage
        for l in range(num_layers):
            nodes_strength[l][acc_prefix] = np.concatenate((nodes_strength[l][acc_prefix], CNet.nodes_strength(l)))
        if processed_files >= maxfiles:
            break
        idx_glob += 1
for l in range(num_layers):
    print("[logger]: Generating plot for layer {}".format(l))
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(nodes_strength[l][acc_prefix]) != 0:
            # Generate PDF
            min_, max_ = np.min(nodes_strength[l][acc_prefix]), np.max(nodes_strength[l][acc_prefix])
            x = np.arange(min_, max_, abs(max_-min_)/1000)
            density = stats.kde.gaussian_kde(nodes_strength[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
            # Collect densities
            nodes_strength_densities += [(x, density)]  # append data and density *function*
    plt.title("{} Nodes Strength LAYER {}".format(dataset, l))
    plt.xlabel("S")
    plt.ylabel("PDF(S)")
    fig_name = "{}_{}_{}_nodes-strength_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
    plt.savefig(saved_images_path + fig_name)
    #plt.show()
    plt.close()
    
# Nodes fluctuation
link_nodes_fluctuation_single_layer = {k:v for (k,v) in zip(['{:4.4f}'.format(r) for r in ranges_accuracy], [np.array([]) for _ in range(num_colors)])}
nodes_fluctuation= [cp.copy(link_nodes_fluctuation_single_layer) for _ in range(num_layers)]  # one dictionary per layer
print("\n[logger]: Generating fluctuations histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    acc_prefix = "{:4.4f}".format(acc)
    extended_acc_prefix = ["{:4.4f}".format(a) for a in np.arange(acc, acc+bins_size, 0.025)]
    files_ = [files_pattern + 'binaccuracy-{}'.format(ap) + '*.npy' for ap in extended_acc_prefix]
    n_files = sum([len(glob.glob(f)) for f in files_])
    print("[logger]: Collecting parameters for {} nets with accuracy {}, with wildcard {}".format(n_files, acc_prefix, files_))
    processed_files, idx_glob, global_files = 0, 0, list(itertools.chain.from_iterable([glob.glob(f) for f in files_]))
    if len(global_files) > 0:  # random shuffle if non-empty
        shuffle(global_files)
    while processed_files != len(global_files):
        processed_files += 1
        file_ = global_files[idx_glob]
        W = np.load(file_, allow_pickle=True)  # load parameters
        if  np.any([np.isnan(w).any() for w in W]):
            continue
        CNet = ComplexNetwork(architecture, num_layers, 0, W, input_size, output_size, strides=None, paddings=None, flatten=False)  # simplify the weights/biases usage
        for l in range(num_layers):
            nodes_fluctuation[l][acc_prefix] = np.concatenate((nodes_fluctuation[l][acc_prefix], CNet.nodes_fluctuation(l)))
        if processed_files >= maxfiles:
            break
        idx_glob += 1
for l in range(num_layers):
    print("[logger]: Generating plot for layer {}".format(l))
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(nodes_fluctuation[l][acc_prefix]) > 1:  # at least two elements are needed for density estimation
            # Generate PDF
            min_, max_ = np.min(nodes_fluctuation[l][acc_prefix]), np.max(nodes_fluctuation[l][acc_prefix])
            x = np.arange(min_, max_, abs(max_-min_)/1000)
            density = stats.kde.gaussian_kde(nodes_fluctuation[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
    plt.title("{} Nodes Fluctuation LAYER {}".format(dataset, l))
    plt.xlabel("Yi")
    plt.ylabel("PDF(Yi)")
    fig_name = "{}_{}_{}_nodes-fluctuation_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
    plt.savefig(saved_images_path + fig_name)
    #plt.show()
    plt.close()

# Plot weights divergences
heatmap_link_weights = np.zeros(shape=(len(ranges_accuracy), len(ranges_accuracy)))
for i in range(len(ranges_accuracy)):
    for j in range(len(ranges_accuracy)):
        if j > i:
            break
        d1, d2 = link_weights_densities[i][0], link_weights_densities[j][0]  # collect data
        P, Q = link_weights_densities[i][1], link_weights_densities[j][1]  # collect distributions
        heatmap_link_weights[i,j] = heatmap_link_weights[j,i] = shannon_divergence(d1, d2, distr=(P, Q))
# Prevent inf in the heatmap
max_divergence = heatmap_link_weights[heatmap_link_weights!=np.inf].max()
heatmap_link_weights[heatmap_link_weights==np.inf] = max_divergence+1
plt.title("{} Shannon Divergence for the Link Weights (accuracies on the axis)".format(dataset))
plt.imshow(heatmap_link_weights, cmap='hot', interpolation='nearest')
fig_name = "{}_{}_{}_heatmap_link-weights_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
plt.savefig(saved_images_path + fig_name)
#plt.show()
plt.close()
print("Shannon divergence (layer by layer) (Matrix)\n ", heatmap_link_weights)
np.save(saved_images_path + "{}_{}_{}_heatmap_link-weights_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, '.npy'), heatmap_link_weights)

# Plot nodes strength divergences
heatmap_nodes_strength = np.zeros(shape=(len(ranges_accuracy), len(ranges_accuracy)))
for i in range(len(ranges_accuracy)):
    for j in range(len(ranges_accuracy)):
        if j > i:
            break
        d1, d2 = nodes_strength_densities[i][0], nodes_strength_densities[j][0]  # collect data
        P, Q = nodes_strength_densities[i][1], nodes_strength_densities[j][1]  # collect distributions
        heatmap_nodes_strength[i,j] = heatmap_nodes_strength[j,i] = shannon_divergence(d1, d2, distr=(P, Q))
# Prevent inf in the heatmap
max_divergence = heatmap_nodes_strength[heatmap_nodes_strength!=np.inf].max()
heatmap_nodes_strength[heatmap_nodes_strength==np.inf] = max_divergence+1
plt.title("{} Shannon Divergence for the Nodes Strength (accuracies on the axis)".format(dataset))
plt.imshow(heatmap_nodes_strength, cmap='hot', interpolation='nearest')
fig_name = "{}_{}_{}_heatmap_nodes-strength_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, img_format)
plt.savefig(saved_images_path + fig_name)
#plt.show()
plt.close()
print("Shannon divergence (layer by layer) (Matrix)\n ", heatmap_nodes_strength)
np.save(saved_images_path + "{}_{}_{}_heatmap_nodes-strength_init-{}_support-{}_layer-{}{}".format(dataset, netsize, architecture, (init if init!='*' else 'any'), scaling_factor, l, '.npy'), heatmap_nodes_strength)
