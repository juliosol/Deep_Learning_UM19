from builtins import range
import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    N, Din = x.shape
    a1 = np.matmul(x, w) + b
    out = a1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    N, Dout = np.shape(dout)

    dx = np.matmul(dout, np.transpose(w))

    dw = np.matmul(np.transpose(x), dout)

    db = np.matmul(np.ones((1, N)), dout)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    #print(np.shape(x))
    out = np.where(x < 0, 0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    #print(np.shape(dout))
    #dx = np.where(dout > 0, 0, 1)
    dx = np.copy(x)
    dx[dx >= 0] = 1
    dx[dx < 0] = 0
    dx = np.multiply(dx,dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        var_eps = sample_var + eps
        x_hat = (x - sample_mean)/np.sqrt(var_eps)
        y = gamma*x_hat + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        out = y
        cache = [x, sample_mean, var_eps, x_hat, gamma, beta]
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        mean_test = bn_param['running_mean']
        test_var = bn_param['running_var']
        x_hat = (x - mean_test)/np.sqrt((test_var + eps))
        y = gamma*x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, sample_mean, var_eps, x_hat, gamma, beta = cache
    N, D = x.shape
    dx_hat = dout * gamma
    d_var = np.sum(dx_hat*(x - sample_mean)*(-1/2)*np.power(var_eps, -3/2), axis=0)
    d_mean = np.sum(dx_hat*(-1*np.power(var_eps, -1/2)), axis=0) + d_var*np.sum(-2*(x - sample_mean), axis=0)/N
    dx = dx_hat*np.power(var_eps, -1/2) + d_var*(2*(x - sample_mean))/N + d_mean*1/N
    #print(np.shape(dx))
    dgamma = np.sum(dout*x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        #mask = np.random.random_integers(0,1,size=x.shape)
        mask = np.random.uniform(0, 2, size=x.shape)
        mask = np.where(mask <=p, 1, 0)
        #mask = np.random.choice([0, 1], x.shape, 1-p)
        out = x * mask
        #out = np.multiply(x, mask)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = np.random.uniform(0, 2, size=x.shape)
        mask = np.where(mask <=p, 1, 0)
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode'] ## What is this mode used for?

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        #mask = np.random.choice([0, 1], x.shape, 1-p)
        dx = np.multiply(dout, mask)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # Now that we have the functions for im2col, we can get forward pass convolution
    # by just multiplying a vector and reshaping.
    F, C, HH, WW = np.shape(w)
    N, C, H, W = np.shape(x)
    H_prime = H - HH + 1
    W_prime = W - WW + 1

    x_cols = im2col(x, HH, WW, 1, padding=0)
    w_col = w.reshape(F, -1)

    out = np.matmul(w_col, x_cols)

    out = out.reshape(N, F, H_prime, W_prime)
    #print("This is the shape of out conv_forward " + str(np.shape(out)))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward

    H_prime = (H - HH + 2*padding)/stride + 1

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    x, w = cache
    F, C, HH, WW = np.shape(w)
    N, C, H, W = np.shape(x)
    H_prime = (H - HH) + 1
    W_prime = W - WW + 1

    x_cols = im2col(x, HH, WW, 1)

    ### Rethink about these ones!!

    dout_reshaped = dout.transpose(0, 1, 2, 3).reshape(F, -1)
    dw = np.matmul(dout_reshaped, np.transpose(x_cols))
    dw = dw.reshape(np.shape(w))

    w_reshape = w.reshape(F,-1)
    dX_col = w_reshape.T @ dout_reshaped

    dx = col2im_indices(dX_col, x, HH, WW, 1, padding=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = np.shape(x)
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    out = None

    H_prime = np.int(1 + (H - pool_height) / stride)
    W_prime = np.int(1 + (W - pool_width) / stride)

    x_reshaped = x.reshape(N*C, 1, H, W)
    x_cols = im2col(x_reshaped, pool_height, pool_width, stride)
    max_position = np.argmax(x_cols, axis=0)
    max_position = max_position.astype(int)
    out = x_cols[max_position, range(max_position.size)]

    out = out.reshape(H_prime, W_prime, N, C).transpose(2, 3, 0, 1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    # Get data back from cache
    x, pool_param = cache

    # Get data from tensor and parameter
    N, C, H, W = np.shape(x)
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    #print(stride)
    N, C, Hd, Wd = np.shape(dout)

    h_out = np.int((H-pool_height)/stride + 1)
    w_out = np.int((W-pool_width)/stride + 1)

    F, C, H_prime, W_prime = np.shape(dout)
    H = (H_prime - 1)*2 + pool_height
    W = (W_prime - 1)*2 + pool_width

    dx = None
    dx = np.zeros(np.shape(x))

    for n in range(N):
        for c in range(C):
            for i in range(Hd):
                for j in range(Wd):
                    height_start = stride * i
                    height_end = stride * i + pool_height
                    width_start = j * stride
                    width_end = j * stride + pool_width
                    x_pool = x[n, c, height_start:height_end, width_start:width_end]
                    mask_max_value = (x_pool == np.max(x_pool))
                    dx[n, c, height_start:height_end, width_start:width_end] += mask_max_value*dout[n, c, i, j]
                    #print("This is mask " + str(mask_max_value))
                    #print(np.shape(dx[n, c, height_start:height_end, width_start:width_end]))
                    #print("This is the product " + str(mask_max_value*dout[n, c, i, j]))
    return dx

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient for binary SVM classification.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  x_sigmoid = 1/(1 + np.exp(-x))
  N = x.shape[0]
  y[y == 0] = -1
  y = y.reshape((-1, 1))
  x = x.reshape((-1, 1))
  loss1 = np.sum(np.maximum(0, 1 - y*x))/N
  loss = np.mean(np.maximum(0, 1 - y*x))
  dx = - y / N
  dx = dx.reshape((-1, 1))
  dx[1 - y*x < 0] = 0
  dx = dx.reshape((-1))

  return loss, dx


def logistic_loss(x, y):
  """
  Computes the loss and gradient for binary classification with logistic
  regression.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  x_sigmoid = 1/(1 + np.exp(-x))
  #print("This is shape for logistic " + str(np.shape(x_sigmoid)))
  #import sys; sys.exit()
  y = np.where(y < 0, 0, y)

  loss = np.mean(-y*np.log(x_sigmoid) - (1-y)*np.log(1-x_sigmoid))
  x_sigmoid = x_sigmoid.reshape(-1)
  dx = (x_sigmoid - y)/np.shape(x)[0]

  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  # Softmax uses the cross entropy loss function and has a
  # linear score function.
  num_samples = y.shape[0]
  num_classes = x.shape[1]
  x_softmax = np.exp(x)/np.sum(np.exp(x), axis=1)[:,None]

  sum = 0
  for k in range(num_classes):
      sum = sum + (y == k)*np.log(x_softmax[:, k])
  loss = np.sum(-sum, axis = 0)/num_samples

  dx = x_softmax
  dx[range(num_samples), y] -= 1
  dx = dx/num_samples

  return loss, dx

#########################################################################
########## ADDED FOR HELP WITH CONV AND MAX POOL FORWARD AND #############
################# BACKWARD PASS #########################################


def im2col_indices(x, pool_height, pool_width, pool_stride, padding=0):
  """
  im2col_indices is a function that will determine where the filter w will act
  according to a stride of 1 and 0 padding on x.
  """
  #F, C, HH, WW = np.shape(w)
  N, C, H, W = np.shape(x)
  HH = pool_height
  WW = pool_width
  stride = pool_stride
  H_prime = int((H + 2 * padding - HH)/stride + 1)
  W_prime = int((W + 2 * padding - WW)/stride + 1)

  # First we compute all the possible combinations for i,j and k (in corr.
  # channel) that our window where we would apply the filter would start.

  # This computes all the possible vectors we could form we elements in
  # the filter and the data matrix using the row position elements.
  # i0 corresponds to the row positions of the filter matrix and i1 corr
  # to the row positions of the data matrix.
  i0 = np.repeat(np.arange(HH), WW)
  i0 = np.tile(i0, C)
  #print("This is i0 " + str(i0))
  #print(np.shape(i0))
  i1 = stride * np.repeat(np.arange(H_prime), W_prime)
  #print("This is i1 " + str(i1))
  #print(np.shape(i1))

  # This computes all the possible vectors we could form we elements in
  # the filter and the data matrix using the column position elements.
  # j0 corresponds to the column positions of the filter matrix and
  # j1 corresponds to the column positions of the data matrix.
  j0 = np.tile(np.arange(WW), HH * C)
  #print("This is j0 " + str(j0))
  #print(np.shape(j0))
  j1 = stride * np.tile(np.arange(W_prime), H_prime)
  #print("This is j1 " + str(j1))
  #print(np.shape(j1))

  # Now we perform all the possible combinations of entries in one matrix
  # as follows:
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  #print("This is shape of i " + str(np.shape(i)))
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  #print("This is shpae of j " + str(np.shape(j)))

  # This computes the possible positions with respect to the corresponding
  # channel
  k = np.tile(np.arange(C), WW*HH).reshape(-1,1)

  #print(k)
  #print("This is i " + str(np.shape(i)))
  #print("This is j " + str(np.shape(j)))

  return k, i, j

def im2col(x, pool_height, pool_width, pool_stride, padding=0):
    """
    im2col is a function that receives a a tensor of data and a tensor of
    filters as input and will convert the tensor of images into a
    collection of matrices so that we just apply the filter we want
    so many for loops
    """
    #F, C, HH, WW = np.shape(w)
    N, C, H, W = np.shape(x)
    HH = pool_height
    WW = pool_width
    H_prime = int((H + 2 * padding - HH)/pool_stride + 1)
    W_prime = int((W + 2 * padding - WW)/pool_stride + 1)

    p = padding
    x_padded = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')

    ## Now we use the im2col_indices function to compute the convoluted matrix
    # by just doing a vector matrix multiplication and reshaping the result.

    # First we get the indices of the column entries for our vectorize matrix
    k, i, j = im2col_indices(x, HH, WW, pool_stride)

    #print(type(k))
    k = k.astype(int)
    i = i.astype(int)
    j = j.astype(int)

    cols = x[:, k, i, j]
    #print(np.shape(x))
    #print(np.shape(cols))
    #print(x[0,0,:,:])
    C = np.shape(x)[1]
    cols = cols.transpose(1, 2, 0).reshape(HH * WW * C, -1)

    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, stride=1, padding=0):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = np.shape(x_shape)
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = im2col_indices(x_shape, field_height, field_width, stride, padding=0)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
