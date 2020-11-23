import numpy as np

class Convolution(object):
    """
    Calculate strength for CNN2D layers.
    Convolutional layers are assumed to have weights of shape <h,w,ci,co> where h,w are the width and length, 
    respectively while ci is the number of input channels (i.e., if we have an RGB image, this will have value 3),
    co is the number of channels of the kernel, and so the number of input channels ci of the successive layer.
    Example of usage for MNIST images (shape is (28,28,1)):
    w, b = model.layers[0].get_weights()
    cl = Convolution((28,28,1), w, b, stride=1, padding=1)
    """
    def __init__(self, input_shape, kernel, bias, stride, padding):
        self.input_h, self.input_w, self.input_channels = input_shape[0], input_shape[1], input_shape[2]
        self.kernel, self.bias = kernel, bias
        self.kernel_h, self.kernel_w, self.kernel_input_channels, self.kernel_channels = kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]
        assert self.input_channels == self.kernel_input_channels, print("Kernel's channels and input channels must be equal, but they are respectively {} and {}".format(self.input_channels, self.kernel_input_channels))
        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        elif isinstance(stride, tuple) or isinstance(stride, list):
            self.stride_h, self.stride_w = stride[0], stride[1]
        else:
            raise NotImplementedError("{} is not a valid value for stride.".format(padding))        
        if isinstance(padding, int):
            self.padding_h = self.padding_w = padding
        elif isinstance(padding, tuple) or isinstance(padding, list):
            self.padding_h, self.padding_w = padding[0], padding[1]
        else:
            raise NotImplementedError("{} is not a valid value for padding.".format(padding))
        self.output_shape = (int((self.input_h+2*self.padding_h-self.kernel_h)/self.stride_h)+1, 
                             int((self.input_w+2*self.padding_w-self.kernel_w)/self.stride_w)+1, 
                             self.kernel_channels)
        self.input_strenght, self.output_strength = self.get_strengths()

    def get_strengths(self):
        input_strength = np.zeros(shape=(self.input_channels, self.kernel_channels, int(self.input_h/self.stride_h), int(self.input_w/self.stride_w)))
        output_strength = np.zeros(shape=(self.output_shape[2], self.output_shape[1], self.output_shape[0]))
        # Input strenght
        for ic in range(self.input_channels):
            for c in range(self.kernel_channels):
                for i in range(0, self.input_h-self.kernel_h+1, self.stride_h):
                    for j in range(0, self.input_w-self.kernel_w+1, self.stride_w):
                        for n in range(self.kernel_h):
                            for m in range(self.kernel_w):
                                input_strength[ic,c,j+m,i+n] += self.kernel[n,m,ic,c]        
        # Output strenght
        for c in range(self.kernel_channels):
            for i in range(0, self.input_h-self.kernel_h+1, self.stride_h):
                for j in range(0, self.input_w-self.kernel_w+1, self.stride_w):
                    output_strength[c,j,i] = np.sum(self.kernel[:,:,:,c])
                    output_strength[c,j,i] += self.bias[c]
        return input_strength, output_strength
        
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
model = keras.Sequential( 
                        [ 
                        keras.Input(shape=input_shape), 
                        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), 
                        #layers.MaxPooling2D(pool_size=(2, 2)), 
                        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), 
                        #layers.MaxPooling2D(pool_size=(2, 2)), 
                        layers.Flatten(), 
                        #layers.Dropout(0.5), 
                        layers.Dense(num_classes, activation="softmax"), 
                        ] 
                        ) 
model.summary() 

from Convolution import Convolution
c = Convolution(input_shape, model.layers[0].get_weights()[0]/model.layers[0].get_weights()[0], model.layers[0].get_weights()[1]/model.layers[0].get_weights()[1] , stride=1, padding=1)
"""