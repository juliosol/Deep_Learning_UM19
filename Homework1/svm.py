import numpy as np
from solver import *
from layers import *
import pickle
from logistic import *
#import pandas as pd

class SVM(object):
  """
  A binary SVM classifier with optional hidden layers.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    if hidden_dim is None:
      W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, 1))
      b1 = np.zeros(1)
      self.params['b1'] = b1
      self.params['W1'] = W1

    else:
      W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dim))
      W2 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, 1))
      b1 = np.zeros((hidden_dim))
      b2 = 0
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
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the classification
    score for X[i].

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    #y = np.where(y == 0, -1, y)
    N, D = np.shape(X)
    W1 = self.params['W1']
    b1 = self.params['b1']

    #print(self.params['W1'])
    if 'W2' not in self.params:
      out1, cache1 = fc_forward(X, W1, b1)
      p_1 = 1/(1 + np.exp(-out1))
      #scores = 1/(1 + np.exp(-out1)).flatten()
      scores = np.concatenate([1-p_1, p_1], axis=1)
      #scores.shape(-1, 1)

    else:
      W2 = self.params['W2']
      b2 = self.params['b2']
      out1, cache1 = fc_forward(X, W1, b1)
      out2, cache2 = fc_forward(out1, W2, b2)
      #z1 = np.dot(X, W1) + b1
      #z2 = np.dot(z1, W2) + b2
      #scores1 = 1/(1 + np.exp(-out1))
      p_1 = 1/(1 + np.exp(-out2))
      scores = np.concatenate([1-p_1, p_1], axis=1)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    if 'W2' not in self.params:
      x, W1, b1 = cache1
      loss, dout = svm_loss(out1, y)
      #print("This is loss for svm " + str(loss))
      #print("This is dout for svm " + str(dout))
      #import sys; sys.exit()
      loss = loss + 1/2*self.reg*np.linalg.norm(W1)**2
      dout = dout.reshape(-1,1)
      dx1, dw1, db1 = fc_backward(dout, cache1)

      grads['W1'] = dw1
      grads['b1'] = db1.reshape(-1)

    else:
      x, W2, b2 = cache2
      x, W1, b1 = cache1
      loss2, dout2 = svm_loss(out2, y)
      #loss1, dout1 = logistic_loss(out1, y)
      loss2 = loss2 + 1/2*self.reg*np.linalg.norm(W2)**2
      dout2 = dout2.reshape(-1, 1)
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
