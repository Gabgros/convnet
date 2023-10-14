import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        (N, C, H, W) = x.shape

        H_out = (H - self.kernel_size)//self.stride + 1
        W_out = (W - self.kernel_size)//self.stride + 1

        out = np.zeros((N, C, H_out, W_out))

        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                max_values = np.max(x[:, :, i:i+self.kernel_size, j:j+self.kernel_size], axis=(2, 3))
                out[:, :, i//self.stride, j//self.stride] = max_values
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        (N, C, H, W) = x.shape
        self.dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        pool_window = x[n, c,
                                      i*self.stride: i*self.stride+self.kernel_size,
                                      j*self.stride: j*self.stride+self.kernel_size]
                        max_indices_window = np.unravel_index(indices= np.argmax(pool_window),
                                                              shape= pool_window.shape)
                        max_indices_x = [i*self.stride+max_indices_window[0],
                                         j*self.stride+max_indices_window[1]]
                        self.dx[n, c, max_indices_x[0], max_indices_x[1]] += dout[n, c, i, j]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
