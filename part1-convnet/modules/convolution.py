import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        x_pad = np.pad(x,((0, 0),(0, 0),(self.padding, self.padding),(self.padding, self.padding)))
        (N, C, H, W) = x.shape

        H_out = (H - self.kernel_size + 2*self.padding)//self.stride + 1
        W_out = (W - self.kernel_size + 2*self.padding)//self.stride + 1

        out = np.zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):
            for c_out in range(self.out_channels):
                for i in range(0, H_out * self.stride, self.stride):
                    for j in range(0, W_out * self.stride, self.stride):
                        conv_window = x_pad[n, :, i:i+self.kernel_size, j:j+self.kernel_size]
                        out[n, c_out, i//self.stride, j//self.stride] = (
                                np.sum(conv_window * self.weight[c_out, :, :, :]) + self.bias[c_out])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        (N, C, H, W) = x.shape
        H_out = (H - self.kernel_size + 2 * self.padding)//self.stride + 1
        W_out = (W - self.kernel_size + 2 * self.padding)//self.stride + 1
        x_pad = np.pad(x,((0, 0),(0, 0),(self.padding, self.padding),(self.padding, self.padding)))

        tempo_dx = np.zeros_like(x_pad)
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

        for n in range(N):
            for c_out in range(self.out_channels):
                for i in range(0, H_out * self.stride, self.stride):
                    for j in range(0, W_out * self.stride, self.stride):
                        conv_window = x_pad[n, :, i:i+self.kernel_size, j:j+self.kernel_size]

                        self.dw[c_out, :, :, :] += conv_window * dout[n, c_out, i//self.stride, j//self.stride]
                        self.db[c_out] += dout[n, c_out, i//self.stride, j//self.stride]
                        tempo_dx[n, :, i:i+self.kernel_size, j:j+self.kernel_size] += (
                                self.weight[c_out] * dout[n, c_out, i//self.stride, j//self.stride])

        self.dx = np.copy(tempo_dx[:, :, self.padding:-self.padding, self.padding:-self.padding])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################