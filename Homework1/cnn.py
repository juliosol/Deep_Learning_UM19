import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:

  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #if hidden_dim is None:
    #  W1 = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, num_classes))
    #  b1 = np.zeros(num_classes)
    #  self.params['b1'] = b1
    #  self.params['W1'] = W1

    #if hidden_dim == 2:
    #  W1 = np.random.normal(loc=0, scale=weight_scale, size=(num_filters, input_dim[1], filter_size, filter_size))
    #  W2 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
    #  b1 = np.zeros(1)
    #  b2 = np.zeros(num_classes)
    #  self.params['b1'] = b1
    #  self.params['W1'] = W1
    #  self.params['W2'] = W2
    #  self.params['b2'] = b2

    #else:
    W1 = np.random.normal(loc=0, scale=weight_scale, size=(num_filters, input_dim[1], filter_size, filter_size))
    W2 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
    W3 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
    b1 = np.zeros(1)
    b2 = np.zeros((hidden_dim))
    b3 = np.zeros(num_classes)
    self.params['b1'] = b1
    self.params['W1'] = W1
    self.params['W2'] = W2
    self.params['b2'] = b2
    self.params['W3'] = W3
    self.params['b3'] = b3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    #if hidden_dim == 2:
    #  out1, cache1 = conv_forward(X, W1) + b1

    #  out_relu, cache_relu = relu_forward(out1)

    #  out_max_pool, cache_x_poool = max_pool_forward(out_relu, pool_param)

    #  out2_fc_forward, cache2 = fc_forward(out_max_pool, W2, b2)

      #out_relu_fc_forward, cache_relu_fc_forward = relu_forward(out_fc_forward)

      #out3_fc_forward, cache3 = fc_forward(out_relu_fc_forward, W3, b3)

    #  exp_column = np.exp(out2_fc_forward)
    #  scores = exp_column/np.sum(exp_column, axis=1, keepdims=True)

    #else:
    C, H, W = np.shape(X)
    X_rshp = X.reshape((1, C, H, W))
    out1, cache1 = conv_forward(X_rshp, W1) + b1

    out_relu, cache_relu = relu_forward(out1)

    out_max_pool, cache_x_poool = max_pool_forward(out_relu, pool_param)

    out_max_pool = out_max_pool.reshape(np.shape(out_max_pool)[0], -1)

    out2_fc_forward, cache2 = fc_forward(out_max_pool, W2, b2)

    out_relu_fc_forward, cache_relu_fc_forward = relu_forward(out_fc_forward)

    out3_fc_forward, cache3 = fc_forward(out_relu_fc_forward, W3, b3)

    exp_column = np.exp(out2_fc_forward)
    scores = exp_column/np.sum(exp_column, axis=1, keepdims=True)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #if hidden_dim == 2:
    #  x, W2, b2 = cache2
    #  x, W1, b1 = cache1

    #  loss2, dout2 = softmax_loss(out2_fc_forward, y)
    #  loss2 = loss2 + 1/2*self.reg*np.linalg.norm(W2)**2
    #  loss = loss + loss2
    #  dout2 = dout2.reshape(-1, 1)

      #dx2, dw2, db2 = fc_backward(dout2, cache2)
      #dx_relu_max_pool = relu_backward(dx2, cache_relu_fc_forward)

    #  dx2, dw2, db2 = fc_backward(dx_relu_max_pool, cache2)

    #   dx_max_pool = max_pool_backward(dx2, cache_x_pool)

    #  dx_relu = relu_backward(dx_max_pool, cache_relu)

    #  dx1, dw1 = conv_backward(dx_max_pool, cache1)

    #  db1 = np.zeros()

    #  grads['W1'] = dw1
    #  grads['b1'] = db1
    #  grads['W2'] = dw2
    #  grads['b2'] = db2


    #else:
    x, W3, b3 = cache3
    x, W2, b2 = cache2
    x, W1, b1 = cache1

    loss3, dout3 = softmax_loss(out3_fc_forward, y)
    loss3 = loss3 + 1/2*self.reg*np.linalg.norm(W3)**2
    loss = loss + loss3
    dout3 = dout3.reshape(-1, 1)

    dx3, dw3, db3 = fc_backward(dout3, cache3)

    dx_relu_max_pool = relu_backward(dx3, cache_relu_fc_forward)

    dx2, dw2, db2 = fc_backward(dx_relu_max_pool, cache2)

    dx2 = dx2.reshape(np.shape(out_max_pool))

    dx_max_pool = max_pool_backward(dx2, cache_x_pool)

    dx_relu = relu_backward(dx_max_pool, cache_relu)

    dx1, dw1 = conv_backward(dx_max_pool, cache1)
    db1 = 0

    grads['W1'] = dw1
    grads['b1'] = db1
    grads['W2'] = dw2
    grads['b2'] = db2
    grads['W3'] = dw3
    grads['b3'] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
