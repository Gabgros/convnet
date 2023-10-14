from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight
        self.velocity = []
        for idx, m in enumerate(model.modules):
            self.velocity.append([0]*2)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                self.velocity[idx][0] = (self.momentum * self.velocity[idx][0]
                                         - self.learning_rate * m.dw)
                m.weight = m.weight + self.velocity[idx][0]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                self.velocity[idx][1] = (self.momentum * self.velocity[idx][1]
                                         - self.learning_rate * m.db)
                m.bias = m.bias + self.velocity[idx][1]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################