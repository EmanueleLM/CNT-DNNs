# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:43:58 2020

@author: Emanuele
"""
from Convolution import Convolution

def ComplexNetwork(architecture, num_layers, num_conv_layers, weights, input_shape, output_size, strides, paddings, flatten=True):
    if architecture == 'fc':
        return __FCLayer('fc', num_layers, weights, input_shape, output_size, flatten)  # ignore the number of convolutional layers.
    elif architecture == 'cnn':
        return __CNNLayer('cnn', num_layers, num_conv_layers, weights, input_shape, output_size, strides, paddings, flatten)
    elif architecture == 'rnn':
        pass
    elif architecture == 'attention':
        pass
    else:
        raise NotImplementedError("{} architecture does not exist (use fc, cnn, rnn or attention)".format(architecture))

class __Network(object):
    def __init__(self, architecture, num_layers, weights, input_shape, output_size, strides, paddings, flatten):
        self.architecture = None
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.output_size = output_size
        self.strides = strides
        self.paddings = paddings
        
class __FCLayer(__Network):
    def __init__(self, architecture, num_layers, weights, input_shape, output_size, flatten=True):
        super().__init__(architecture, num_layers, weights, input_shape, output_size, None, None, flatten)
        self.architecture = 'fc'
        self.weights = []
        self.biases = []
        # Collect weights and biases for each layer
        for i in range(0, len(weights), 2):
            self.weights.append((weights[i].flatten() if flatten else weights[i]))
            self.biases.append((weights[i+1].flatten() if flatten else weights[i+1]))
    
    def link_weights(self, layer):
        return (self.weights[layer], self.biases[layer])
            
    def nodes_strength(self, layer):
        if layer == 0:
            strength = self.weights[0].sum(axis=-1) + 1
        # Last layer
        elif layer == self.num_layers-1:
            strength = self.weights[-1].sum(axis=0) + 1 + self.biases[-1]
        else:
            s_in = self.weights[layer].sum(axis=-1) 
            s_out = self.weights[layer-1].sum(axis=0) + self.biases[layer-1]
            strength = s_in + s_out
        return strength
    
    def nodes_fluctuation(self, layer):
        fluctuation = self.nodes_strength(layer).std()
        return [fluctuation]  # enclose the value (a scalar) in a list

class __CNNLayer(__Network):
    # Strong assumption: convolutional layers always preceed in block fc layers.
    def __init__(self, architecture, num_layers, num_conv_layers, weights, input_shape, output_size, strides, paddings, flatten=True):
        super().__init__(architecture, num_layers, weights, input_shape, output_size, strides, paddings, flatten)
        self.architecture = 'cnn'
        self.weights = []
        self.biases = []
        self.num_conv_layers = num_conv_layers
        self.stride = []
        self.padding = []
        self.input_shape_layer = []
        # Collect weights and biases for each layer
        for n,i in enumerate(range(0, len(weights), 2)):
            self.weights.append((weights[i].flatten() if flatten else weights[i]))
            self.biases.append((weights[i+1].flatten() if flatten else weights[i+1]))
            if n < self.num_conv_layers:
                self.stride.append(strides[n])
                self.padding.append(paddings[n])
        # Calculate input shape at each layer
        self.input_shape_layer.append(self.input_shape)
        for l in range(1,self.num_conv_layers):
            h = int((self.input_shape_layer[l-1][0]-self.weights[l].shape[0]+2*self.paddings[l])/self.strides[l])+1
            w = int((self.input_shape_layer[l-1][1]-self.weights[l].shape[1]+2*self.paddings[l])/self.strides[l])+1
            c = self.weights[l].shape[-1]
            self.input_shape_layer.append([h,w,c])
        
    def link_weights(self, layer):
        return (self.weights[layer], self.biases[layer])
            
    def nodes_strength(self, layer):
        """
        Naming: s_in is the strength of the incoming weights, s_out of the outcoming, wrt a layer of neurons.
        Strength is calculated for each layer of neurons.
        """
        # First (conv.) layer
        if layer == 0:
            s_in = 1
            s_out = Convolution(self.input_shape_layer[0], self.weights[0], self.biases[0], self.stride[0], self.padding[0]).input_strength
        # Cnn layer, previous is also cnn layer
        elif self.num_conv_layers > layer > 0:
            s_in = Convolution(self.input_shape_layer[layer-1], self.weights[layer-1], self.biases[layer-1], self.stride[layer-1], self.padding[layer-1]).output_strength
            s_out = Convolution(self.input_shape_layer[layer], self.weights[layer], self.biases[layer], self.stride[layer], self.padding[layer]).input_strength
        # First fc layer, connect with last convolutional
        elif layer == self.num_conv_layers:
            s_in = Convolution(self.input_shape_layer[layer-1], self.weights[layer-1], self.biases[layer-1], self.stride[layer-1], self.padding[layer-1]).output_strength
            s_out = self.weights[layer].sum(axis=-1)
        # Last layer
        elif layer == self.num_layers:
            s_in = self.weights[layer-1].sum(axis=0) + self.biases[layer-1]
            s_out = 1
        # Any fc layer, except the last one
        else:
            s_in = self.weights[layer-1].sum(axis=0) + self.biases[layer-1]
            s_out = self.weights[layer].sum(axis=1) 
        print("[logger-DEBUG]: Strength of layer {}".format(layer))
        print("\t Shape of weights: {}".format(self.weights[layer].shape if layer < self.num_layers else None))
        print("\t S_in shape: {}".format((1 if isinstance(s_in, int) else s_in.shape)))
        print("\t S_out shape: {}".format((1 if isinstance(s_out, int) else s_out.shape)))
        # Flatten before return
        s_in = (1 if isinstance(s_in, int) else s_in.flatten())
        s_out = (1 if isinstance(s_out, int) else s_out.flatten())
        return s_in + s_out

    def nodes_fluctuation(self, layer):
        fluctuation = self.nodes_strength(layer).std()
        return [fluctuation]  # enclose the value (a scalar) in a list  
