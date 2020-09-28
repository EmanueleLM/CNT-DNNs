# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:43:58 2020

@author: Emanuele
"""

def ComplexNetwork(architecture, num_layers, weights, input_size, output_size, flatten=True):
    if architecture == 'fc':
        return __Network('fc', num_layers, weights, input_size, output_size, flatten)
    elif architecture == 'cnn':
        pass
    elif architecture == 'rnn':
        pass
    elif architecture == 'attention':
        pass
    else:
        raise NotImplementedError("{} architecture does not exist (use fc, cnn, rnn or attention)".format(architecture))

class __Network(object):
    def __init__(self, architecture, num_layers, weights, input_size, output_size, flatten):
        self.architecture = None
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
class __FCLayer(__Network):
    def __init__(self, architecture, num_layers, weights, input_size, output_size, flatten=True):
        super(__FCLayer, self).__init__()
        self.architecture = 'fc'
        self.weights = []
        self.biases = []
        # Collect weights and biases for each layer
        for i in range(0, 2*num_layers, 2):
            self.weights.append((weights[i].flatten() if flatten else weights[i]))
            self.biases.append((weights[i+1].flatten() if flatten else weights[i+1]))
            
        def nodes_strength(self, layer):
            if layer == 0:
                strength = self.weights[0].sum(axis=-1) + 1
            elif layer == self.num_layers:
                strength = self.weights[-1].sum(axis=0) + 1 + self.biases[-1]
            else:
                s_in = self.weights[layer].sum(axis=-1) 
                s_out = self.weights[layer-1].sum(axis=0) + self.biases[layer-1]
                strength = s_in + s_out
            return strength
        
        def nodes_fluctuation(self, layer):
            fluctuation = self.nodes_strength(layer).std()
            return fluctuation   
        