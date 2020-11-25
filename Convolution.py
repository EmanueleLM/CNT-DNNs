import numpy as np

class Convolution(object):
    """
    Calculate strength for CNN2D layers.
    Convolutional layers are assumed to have weights of shape <h,w,ci,co> where h,w are the width and length, 
    respectively while ci is the number of input channels (i.e., if we have an RGB image, this will have value 3),
    co is the number of channels of the kernel, and so the number of input channels ci of the successive layer.
    Example of usage for MNIST images (shape is (28,28,1)):
    height = width = 3; input_channels = 1, output_channels = 32
    w, b = np.ones(shape=(height, width, input_channles, output_channels))
    cl = Convolution((28,28,1), w, b, stride=1, padding=1)
    """
    def __init__(self, input_shape, kernel, bias, stride, padding):
        """
        Input:
            input_shape:tuple/list
                shape of the input. It is a 3D tensor, order of the dimensions is (height, width, input_channels)
            kernel:np.array
                weights used for the convolution. It is a 4D tensor, order of the dimensions is (height, width, input_channels, kernel_channels)
            bias:np.array
                biases of the convolutional layer. It is a monodimensional vector, whose dimension is (output_channels,)
            stride:int/tuple/list
                stridesfor each dimension. It is either an integer (i.e., same stride is applied element-wise) or a list with two entries, (height and width)
            padding:int/tuple/list
                number of padding pixels added to each dimension. It is either an integer (i.e., same padding is applied element-wise) or a list with two entries, (height and width)
        Output:
            an object Covolution, whose elements input_strength and output_strength are the nodes strength as the result of the convolution.
        """
        # Padding
        if isinstance(padding, int):
            self.padding_h = self.padding_w = padding
        elif isinstance(padding, tuple) or isinstance(padding, list):
            self.padding_h, self.padding_w = padding[0], padding[1]
        else:
            raise NotImplementedError("{} is not a valid value for padding.".format(padding))
        # Padded input and kernels
        self.input_h, self.input_w, self.input_channels = input_shape[0]+2*self.padding_h, input_shape[1]+2*self.padding_w, input_shape[2]
        self.kernel, self.bias = kernel, bias
        self.kernel_h, self.kernel_w, self.kernel_input_channels, self.kernel_channels = kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]
        assert self.input_channels == self.kernel_input_channels, print("Kernel's channels and input channels must be equal, but they are respectively {} and {}".format(self.input_channels, self.kernel_input_channels))
        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        elif isinstance(stride, tuple) or isinstance(stride, list):
            self.stride_h, self.stride_w = stride[0], stride[1]
        else:
            raise NotImplementedError("{} is not a valid value for stride.".format(padding))        
        self.output_shape = (int((self.input_h-self.kernel_h)/self.stride_h)+1, 
                             int((self.input_w-self.kernel_w)/self.stride_w)+1, 
                             self.kernel_channels)
        self.input_strength, self.output_strength = self.get_strengths()

    def get_strengths(self):
        input_strength = np.zeros(shape=(self.input_channels, self.input_h, self.input_w))
        output_strength = np.zeros(shape=(self.output_shape[2], self.output_shape[0], self.output_shape[1]))
        # Input strenght
        for ic in range(self.input_channels):
            for c in range(self.kernel_channels):
                for i in range(0, self.input_h-self.kernel_h+1, self.stride_h):
                    for j in range(0, self.input_w-self.kernel_w+1, self.stride_w):
                        for n in range(self.kernel_h):
                            for m in range(self.kernel_w):
                                input_strength[ic,i+n,j+m] += self.kernel[n,m,ic,c]
        # Output strenght
        for c in range(self.kernel_channels):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    output_strength[c,i,j] = np.sum(self.kernel[:,:,:,c])
                    output_strength[c,i,j] += self.bias[c]
        border_h, border_w = self.input_h-self.padding_h, self.input_w-self.padding_w
        return input_strength[:,self.padding_h:border_h,self.padding_w:border_w], output_strength[:,self.padding_h:border_h,self.padding_w:border_w]

"""
# Example
from Convolution import Convolution
height = width = 3; input_channels = 3; output_channels = 10
input_shape = (32, 32, input_channels)  # mimic an MNIST image
kernel, bias = np.ones(shape=(3,3,3,10)).reshape(height,width,input_channels,output_channels), np.zeros(shape=(10,))
c = Convolution(input_shape, kernel, bias, stride=1, padding=1)
print("Input strenghts:")
print(c.input_strength)
print("\nOutput strenghts:")
print(c.output_strength)
"""