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

args = parser.parse_args()
architecture = args.architecture
dataset = args.dataset
num_layers = args.num_layers
bins_size = args.bins_size
scaling_factor = args.scale
init = args.init_method

ranges_accuracy = np.arange(0., 1.0, bins_size)
input_size, output_size = (28*28 if dataset=='MNIST' else 32*32*3), 10
topology = 'fc'
init = ('*' if len(init)==0 else init)
files_pattern = "./weights/{}/{}_{}_*init-{}_support-{}*".format(dataset, dataset, architecture, init, scaling_factor)  # wildcards for architecture and accuracy
saved_images_path = "./results/images/{}/".format(dataset)

# Set colors for plotting (green to red, low to high accuracy)
num_nets = len(glob.glob(files_pattern))
num_colors = len(ranges_accuracy)
red = Color("green")
colors = list(red.range_to(Color("red"), num_colors))

# Link weights
layers_link_weights = {}
link_weights_single_layer = {k:v for (k,v) in zip(['{:4.4f}'.format(r) for r in ranges_accuracy], [np.array([]) for _ in range(num_colors)])}
link_weights = [cp.copy(link_weights_single_layer) for _ in range(num_layers)]  #♣ one dictionary per layer
print("\n[logger]: Generating weights histogram PDFs and error-bars")
for i, acc in enumerate(ranges_accuracy):
    acc_prefix = "{:4.4f}".format(acc)
    files_ = files_pattern + 'binaccuracy-{}'.format(acc_prefix) + '*.npy'
    n_files = len(glob.glob(files_))
    print("[logger]: {} nets with accuracy{} with wildcard {}".format(n_files, acc_prefix, files_))
    for file_ in glob.glob(files_):
        W = np.load(file_, allow_pickle=True)  # load parameters
        nn_layer = ComplexNetwork(architecture, num_layers, W, input_size, output_size, flatten=True)  # simplify the weights/biases usage
        for l in range(num_layers):
            link_weights[l][acc_prefix] = np.concatenate((link_weights[l][acc_prefix], nn_layer.weights[l], nn_layer.biases[l]))
for l in range(num_layers):
    for i, acc in enumerate(ranges_accuracy):
        acc_prefix = "{:4.4f}".format(acc)
        if len(link_weights[l][acc_prefix]) != 0:
            min_, max_ = np.min(link_weights[l][acc_prefix]), np.max(link_weights[l][acc_prefix])
            x = np.arange(min_, max_, .001)
            density = stats.kde.gaussian_kde(link_weights[l][acc_prefix])
            plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
    plt.title("{} Link Weights LAYER {}".format(dataset, l))
    plt.xlabel("W")
    plt.ylabel("PDF(W)")
    plt.show()
    fig_name = "{}_{}_link-weights_init-{}_support-{}_layer-{}.svg".format(dataset, architecture, (init if init!='*' else 'any'), scaling_factor, l)
    plt.savefig(fig_name)
    plt.close()
   
# HISTOGRAMS
# Link weights histogram
print("\n[logger]: Link weights histogram")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))
    print("[logger]: {} files in folder {}".format(n_files, files_))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias) 
    density = stats.kde.gaussian_kde(np.concatenate((total_weights.flatten(), total_bias.flatten())))
    x = np.arange(-1., 1., .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 

# Node strenght input layer
print("\n[logger]: Node strenght input layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_input_layer = total_weights.reshape(n_files, 32, 10).sum(axis=-1)
    density = stats.kde.gaussian_kde(s_input_layer.flatten())
    x = np.arange(s_input_layer.min(), s_input_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_node_strenght_input-layer-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght output layer
print("\n[logger]: Node strenght output layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_output_layer = total_weights.reshape(n_files, 32, 10).sum(axis=1) + total_bias.reshape(n_files, 10,)
    density = stats.kde.gaussian_kde(s_output_layer.flatten())
    x = np.arange(s_output_layer.min(), s_output_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_node_strenght_output-layer-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node disparity input layer
print("\n[logger]: Node disparity input layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        min_w, min_b = np.min(w[-2]), np.min(w[-1])
        min_ = np.abs(min(min_b, min_w))
        total_weights = np.append(w[-2]+min_, total_weights)
        total_bias = np.append(w[-1]+min_, total_bias)
    s_input_layer = total_weights.reshape(n_files, 32, 10).sum(axis=-1)
    Y = np.sum(total_weights.reshape(n_files, 32, 10)**2, axis=-1)/s_input_layer**2
    density = stats.kde.gaussian_kde(Y.flatten())
    x = np.arange(Y.min(), Y.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_node_disparity_input-layer-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node disparity output layer
print("\n[logger]: Node disparity output layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        min_w, min_b = np.min(w[-2]), np.min(w[-1])
        min_ = np.abs(min(min_b, min_w))
        total_weights = np.append(w[-2]+min_, total_weights)
        total_bias = np.append(w[-1]+min_, total_bias)
    s_output_layer = total_weights.reshape(n_files, 32, 10).sum(axis=1) + total_bias.reshape(n_files, 10,)
    Y = (np.sum(total_weights.reshape(n_files, 32, 10)**2, axis=1)+total_bias.reshape(n_files, 10,)**2)/s_output_layer**2
    density = stats.kde.gaussian_kde(Y.flatten())
    x = np.arange(Y.min(), Y.max(), .0001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_node_disparity_output-layer-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Std input layer
print("\n[logger]: Node std input layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_input_layer = total_weights.reshape(n_files, 32, 10).std(axis=-1)
    density = stats.kde.gaussian_kde(s_input_layer.flatten())
    x = np.arange(s_input_layer.min(), s_input_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_node_std_input-layer-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Std output layer
print("\n[logger]: Node std output layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_output_layer = total_weights.reshape(n_files, 32, 10).std(axis=1) + total_bias.reshape(n_files, 10,)
    density = stats.kde.gaussian_kde(s_output_layer.flatten())
    x = np.arange(s_output_layer.min(), s_output_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}histogram_total_node_std_output-layer-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()


# ERRORBARS (SCATTER):
# weights mean and variance
print("\n[logger]: Errorbar mean-variance")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files  
    wb = np.concatenate((weights.flatten(), bias.flatten()))
    plt.errorbar(a, wb.mean(), yerr=wb.std(), fmt='--o', color=str(colors[i]), alpha=1.)
plt.savefig('{}errorbar_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()    

# node strenght input
print("\n[logger]: Errorbar node strength input")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
    weights /= n_files
    s_input_layer = weights.sum(axis=1)
    plt.errorbar(a, s_input_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}errorbar_node-strenght-accuracy({}-{}-step-{})-input.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node strenght output
print("\n[logger]: Errorbar node strength output")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files
    s_output_layer = weights.sum(axis=0) + bias
    plt.errorbar(a, s_output_layer.mean(), yerr=s_output_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}errorbar_node-strenght-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node disparity input
print("\n[logger]: Errorbar node disparity input")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
    weights /= n_files
    weights += np.abs(np.min(weights))
    s_input_layer = weights.sum(axis=1)
    Y = np.sum(weights**2, axis=1)/s_input_layer**2
    plt.errorbar(a, Y.mean(), yerr=Y.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}errorbar_node-disparity-accuracy({}-{}-step-{})-input.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()  

# node disparity output
print("\n[logger]: Errorbar node disparity output")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files
    min_ = np.abs(min(np.min(weights), np.min(bias)))
    weights += min_
    bias += min_
    s_output_layer = weights.sum(axis=0) + bias
    Y = (np.sum(weights**2, axis=0)+bias**2)/s_output_layer**2
    plt.errorbar(a, Y.mean(), yerr=Y.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}errorbar_node-disparity-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 

# std input
print("\n[logger]: Scatter node std input")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
    weights /= n_files
    s_input_layer = weights.std(axis=1)
    plt.scatter(a, s_input_layer.mean(), color=str(colors[i]), alpha=.5)
plt.savefig('{}errorbar_node-std-accuracy({}-{}-step-{})-input.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# std output
print("\n[logger]: Scatter node std output")
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files
    s_output_layer = weights.std(axis=0) + bias
    plt.scatter(a, s_output_layer.mean(), color=str(colors[i]), alpha=.5)
plt.savefig('{}errorbar_node-std-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()
    

"""
# Multimodal fitting of wieghts
from sklearn import mixture

gmix = mixture.GaussianMixture(n_components = 3, covariance_type = "full")
fitted = gmix.fit(np.concatenate((total_weights, total_bias)).reshape(-1,1))
data=np.concatenate((total_weights.flatten(), total_bias.flatten()))
y,x,_=plt.hist(data,1000,alpha=.3,label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)
params = [-0.34266558, 0.01597775**0.5, 580.,
          -0.02447527, 0.01349047**0.5, 700.,
          0.28088089, 0.01203812**0.5, 630.]
plt.plot(x,trimodal(x,*params),color='red',lw=3,label='model')
plt.legend()
"""

""" Fuzzy histograms (5 for each accuracy level)
results_folders = ["0.1-0.125",
                    "0.375-0.4",
                    "0.575-0.6",
                    "0.775-0.8",
                    "0.95-0.975"]

topology = 'fc'  
step = 0.2
bins = 100
num_colors = len(results_folders)
red = Color("green")
colors = list(red.range_to(Color("red"),num_colors))
nr_fuzzies = 5

# HISTOGRAMS
# Link weights histogram
print("\n[logger]: Link weights histogram")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    total_weights, total_bias = np.array([]), np.array([])
    files_ = glob.glob("./results/{}/{}/*.npy".format(topology, acc))
    rr = np.random.randint(0, len(files_), nr_fuzzies)
    for r in rr:
        w = np.load(files_[r], allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias) 
        density = stats.kde.gaussian_kde(np.concatenate((total_weights.flatten(), total_bias.flatten())))
        x = np.arange(-1., 1., .001)
        plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('./images/{}/fuzzy_histograms-{}_total_weights-accuracy({}-{}-step-{}).png'.format(topology,nr_fuzzies,0.1, 1.0, step))
plt.show()
plt.close()
"""
