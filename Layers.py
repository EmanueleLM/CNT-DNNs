# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:43:58 2020

@author: Emanuele
"""

class FCLayer(object):
    def __init__(self, num_layers, weights, flatten=True):
        self.weights = []
        self.biases = []
        for i in range(0, 2*num_layers, 2):
            self.weights.append((weights[i].flatten() if flatten else weights[i]))
            self.biases.append((weights[i+1].flatten() if flatten else weights[i+1]))
            