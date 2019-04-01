import numpy as np
"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################

    h = x @ Wx + prev_h @ Wh + b

    next_h = np.tanh(h)

    cache = [x, prev_h,Wx, Wh, h]

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################

    N, D = np.shape(dnext_h)
    x, prev_h, Wx, Wh, h = cache 
    #print(np.shape(h))
    #print(np.shape(dnext_h))

    next_h = np.tanh(h)
    dtanh =  (1 - np.tanh(h)**2)
    dout = dnext_h * dtanh
    
    dx = dout @ Wx.T

    dprev_h = dout @ Wh.T

    dWx = x.T @ (dout)

    dWh = prev_h.T @ (dout)

    # This derivative will be dout dot product with a vector of ones, but that
    # is the same as just summing along the rows of the matrix dout.
    db = np.sum(dout,axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################

    T, N, D = np.shape(x)
    prev_h = h0
    h = []
    final_cache = []

    for t in range(T):
        curr_x = x[t,:,:]
        next_h, cache = rnn_step_forward(curr_x, prev_h, Wx, Wh, b)
        h.append(next_h)
        final_cache.append(cache)
        prev_h = next_h    

    h = np.array(h)
    cache = np.array(final_cache)
    #print("Size of cache list", len(cache))
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################

    T,N, H = np.shape(dh)
    #print(T)
    #print(N)
    #print(H)
    x, prev_h, Wx, Wh, h = cache[0]
    _, D = np.shape(x)

    
    dx_list = []
    sum_dh = np.zeros((N,H))
    sum_dWx = np.zeros((D, H))
    sum_dWh = np.zeros((H, H))
    sum_db = np.zeros((H,))
    
    dh = np.array(dh)

    for t in range(T-1, -1, -1):
        curr_cache = cache[t]
        dh_partial = dh[t,:,:] + sum_dh
        
        dx, dprev_h, dWx, dWh, db = rnn_step_backward(dh_partial, curr_cache)
        sum_dh = dprev_h
        sum_dWh = sum_dWh + dWh
        sum_dWx = sum_dWx + dWx
        sum_db = sum_db + db
        dx_list.insert(0,dx)

        dh0 = dprev_h
        dWh = sum_dWh
        dWx = sum_dWx
        db = sum_db
        dx = dx_list
   
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################

    N, D = np.shape(x)
    _, H = np.shape(prev_h)

    A = x @ Wx + prev_h @ Wh + b

    f_t = sigmoid(A[:,0:H])
    i_t = sigmoid(A[:,H : 2*H])
    c_tilde_t = np.tanh(A[:,2*H:3*H])
    o_t = sigmoid(A[:, 3*H : 4*H])

    c_t = f_t * prev_c + i_t * c_tilde_t
    h_t = o_t * np.tanh(c_t)

    next_h = h_t
    next_c = c_t
    cache = [f_t, i_t, c_tilde_t, o_t, prev_h, prev_c, next_c, Wx, Wh, x]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db, dprev_h, dprev_c = None, None, None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################

    f_t, i_t, c_tilde_t, o_t, prev_h, prev_c, next_c, Wx, Wh, x = cache
    
    N, H = np.shape(dnext_h)
    D, _ = np.shape(Wx)
    
    Wx_f = Wx[:,0:H]
    Wh_f = Wh[:,0:H]
    #b_f = b[0:H]

    Wx_i = Wx[:,H:2*H]
    Wh_i = Wh[:,H:2*H]
    #b_i = b[H:2*H]

    Wx_c = Wx[:,2*H:3*H]
    Wh_c = Wh[:,2*H:3*H]
    #b_c = b[2*H:3*H]

    Wx_o = Wx[:,3*H:4*H]
    Wh_o = Wh[:,3*H:4*H]
    #b_o = b[3*H:4*H]    

    #f_t = sigmoid(x @ Wx_f + prev_h @ Wh_f + b_f)
    #i_t = sigmoid(x @ Wx_i + prev_h @ Wh_i + b_i)
    #c_tilde_t = np.tanh(x @ Wx_c + prev_h @ Wh_c + b_c)
    #o_t = sigmoid(x @ Wx_o + prev_h @ Wh_o + b_o)

    c_t = f_t * prev_c + i_t * c_tilde_t
    h_t = o_t * np.tanh(c_t)
    dtanh_ct = (1-np.tanh(c_t)**2)

    df = f_t * (1-f_t)
    di = i_t * (1-i_t)
    do = o_t * (1-o_t)
    d_c_tilde_t = (1 - (c_tilde_t**2))

    change_c_h = dnext_h * (o_t * dtanh_ct) + dnext_c

    change_c_h_f = change_c_h * df * prev_c
    change_c_h_i = change_c_h * di * c_tilde_t
    change_c_h_c = change_c_h * d_c_tilde_t * i_t
    change_c_h_o = dnext_h * do * np.tanh(c_t)

    dWx_f = x.T @ change_c_h_f
    dWx_i = x.T @ change_c_h_i
    dWx_c = x.T @ change_c_h_c
    dWx_o = x.T @ change_c_h_o
    dWx = np.concatenate([dWx_f, dWx_i, dWx_c, dWx_o], axis=1)

    dWh_f = prev_h.T @ change_c_h_f
    dWh_i = prev_h.T @ change_c_h_i
    dWh_c = prev_h.T @ change_c_h_c
    dWh_o = prev_h.T @ change_c_h_o
    dWh = np.concatenate([dWh_f, dWh_i, dWh_c, dWh_o], axis=1)

    dc_x_f = change_c_h_f @ Wx_f.T
    dc_x_i = change_c_h_i @ Wx_i.T
    dc_x_c = change_c_h_c @ Wx_c.T
    dc_x_o = change_c_h_o @ Wx_o.T
    dx = dc_x_f + dc_x_i + dc_x_c + dc_x_o

    dprev_h_f = change_c_h_f @ Wh_f.T
    dprev_h_i = change_c_h_i @ Wh_i.T
    dprev_h_c = change_c_h_c @ Wh_c.T
    dprev_h_o = change_c_h_o @ Wh_o.T
    dprev_h = dprev_h_f + dprev_h_i + dprev_h_c + dprev_h_o

    dprev_c = change_c_h * f_t

    db_f = (((change_c_h) * df * prev_c).T @ np.ones((N,1))).reshape(-1)
    db_i = (((change_c_h) * di * c_tilde_t).T @ np.ones((N,1))).reshape(-1)
    db_c = (((change_c_h) * d_c_tilde_t * i_t).T @ np.ones((N,1))).reshape(-1)
    db_o = ((dnext_h * do * np.tanh(c_t)).T@ np.ones((N,1))).reshape(-1)
    db = np.concatenate([db_f, db_i, db_c, db_o])

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return  dx, dprev_h, dprev_c, dWx, dWh, db
    
def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################

    T, N, D = np.shape(x)
    _, H = np.shape(h0)
    h = []
    c_0 = np.zeros((N,H))
    prev_h = h0
    prev_c = c_0
    c = []
    final_cache = []

    for t in range(T):
        curr_x = x[t,:,:]
        next_h, next_c, cache = lstm_step_forward(curr_x, prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h.append(prev_h)
        c.append(prev_c)
        final_cache.append(cache)

    h = np.array(h)
    cache = np.array(final_cache)
    

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################

    f_t, i_t, c_tilde_t, o_t, prev_h, prev_c, next_c, Wx, Wh, x = cache[0]
    T, N, H = np.shape(dh)
    D, _ = np.shape(Wx)

    dx_list = []
    sum_dh = np.zeros((N,H))
    sum_dWx = np.zeros((D,4*H))
    sum_dWh = np.zeros((H,4*H))
    sum_db = np.zeros((4*H,))
    sum_dc = np.zeros((N,H))

    dnext_c = np.zeros(np.shape(prev_c))
    c0 = np.zeros(np.shape(next_c))

    for t in reversed(range(T)):
        dh_partial = dh[t,:,:] + sum_dh
        partial_c = dnext_c + sum_dc 
        
        curr_cache = cache[t]

        dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dh_partial, partial_c, curr_cache)

        dx_list.insert(0, dx)
        sum_dWx = sum_dWx + dWx
        sum_dWh = sum_dWh + dWh
        sum_db = sum_db + db
        sum_dh = dprev_h 
        sum_dc = dprev_c

        dh0 = dprev_h
        dWx = sum_dWx
        dWh = sum_dWh
        db = sum_db
        dx = np.array(dx_list)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W

    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)

    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """

    N,T,D = np.shape(x)
    M = np.shape(b)[0]
    out = []

    for n in range(N):
        curr_out = x[n,:,:] @ w + b.T
        out.append(curr_out)

    #out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out

    out = np.array(out)
    cache = [x, w, b, out]
    return out, cache

def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """

    #x, w, b, out = cache
    #N, T, D = x.shape
    #M = b.shape[0]

    #dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    #dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    #db = dout.sum(axis=(0, 1))



    N, T, M = np.shape(dout)
    x, w, b, out = cache
    D, _ = np.shape(w)

    dx = []
    dw = np.zeros((D,M))
    db = np.zeros(np.shape(b))
    
    for n in range(N):
        curr_dx = np.matmul(dout[n,:,:], np.transpose(w))
        curr_dw = np.transpose(x[n,:,:]) @ dout[n,:,:]
        curr_db = np.ones((1,T)) @ dout[n,:,:]

        dx.append(curr_dx)
        dw = dw + curr_dw
        db = db + curr_db

    db = db.reshape(-1)
    dx = np.array(dx)

    return dx, dw, db

def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range.
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = np.shape(x)
    
    total_loss = 0
    total_dx = []

    for n in range(N):
        temp_y = y[n,:]
        temp_x = x[n,:,:]
        num_samples = temp_y.shape[0]
        num_classes = temp_x.shape[1]
        x_softmax = (np.exp(temp_x)/np.sum(np.exp(temp_x), axis=1)[:,None])

        summation = 0
        for k in range(num_classes):
            summation = summation + mask[n,:] * (temp_y == k) * np.log(x_softmax[:, k])
        loss = np.sum(-summation, axis=0)

        dx = x_softmax
        for k in range(num_classes):
            dx[:,k] = mask[n,:] * (x_softmax[:,k] - (temp_y == k))/N
        total_dx.append(dx)

        total_loss = total_loss + loss

    loss = total_loss/N
    dx = np.array(total_dx)

    return loss, dx

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
    cache = [x, w, b]
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

    #db = np.matmul(np.ones((1, N)), dout)
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

