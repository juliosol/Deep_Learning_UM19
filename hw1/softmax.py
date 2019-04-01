import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        if hidden_dim is None:
            W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, num_classes))
            b1 = np.zeros(num_classes)
            self.params['b1'] = b1
            self.params['W1'] = W1

        else:
            W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dim))
            W2 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
            b1 = np.zeros((hidden_dim))
            b2 = np.zeros(num_classes)
            self.params['b1'] = b1
            self.params['W1'] = W1
            self.params['W2'] = W2
            self.params['b2'] = b2
############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1 = self.params['W1']
        b1 = self.params['b1']

        if 'W2' not in self.params:
            out1, cache1 = fc_forward(X, W1, b1)
            print("This is fc_forward" + str(out1))
            exp_column = np.exp(out1)
            scores = exp_column/np.sum(exp_column, axis=0, keepdims=True)
            print("These are scores " + str(np.sum(exp_column, axis=0, keepdims=True)))
            #import sys; sys.exit()
            #print("These are the scores for 2 layer softmax " + str(scores))


        else:
            W2 = self.params['W2']
            b2 = self.params['b2']
            out1, cache1 = fc_forward(X, W1, b1)
            out2, cache2 = fc_forward(out1, W2, b2)
            #z2 = np.dot(z1, W2) + b2
            #z1 = np.dot(X, W1) + b1
            #scores1 = 1/(1 + np.exp(-out1))
            exp_column = np.exp(out2)
            scores = exp_column/np.sum(exp_column, axis=1, keepdims=True)
            #print("These are the scores for 2 layer softmax " + str(scores))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        if 'W2' not in self.params:
            x, W1, b1 = cache1
            loss, dout = softmax_loss(out1, y)
            loss = loss + 0.5*self.reg*np.linalg.norm(W1)**2
            dx1, dw1, db1 = fc_backward(dout, cache1)
            db1 = db1.reshape(-1,1)
            grads['W1'] = dw1
            grads['b1'] = db1.reshape(-1)
            #print("This is no hidden layer")

        else:
            print("This is hidden layer")
            x, W1, b1 = cache1
            x, W2, b2 = cache2
            loss2, dout2 = softmax_loss(out2, y)
            loss2 = loss2 + 1/2*self.reg*np.linalg.norm(W2 )**2
            dx2, dw2, db2 = fc_backward(dout2, cache2)
            dx1, dw1, db1 = fc_backward(dx2, cache1)
            grads['W1'] = dw1
            grads['b1'] = db1
            grads['W2'] = dw2
            grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
