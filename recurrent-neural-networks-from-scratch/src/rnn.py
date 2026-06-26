import numpy as np
import math

def rnn_cell_forward(x_t, a_prev, parameters):
    """
    Create a function should take the described 3 inputs and output 
    the calculated a_t and y_t for the given batch of inputs.

    Parameters
    ----------
    x_t - This will be a vector equal to the number of features in a 
    sample, for example in our weather prediction example there were
    14 weather measurements each time step, so `n_x = 14` and the input 
    would be vectors of shape `(14,)`
    a_prev - This will be a vector of shape `(n_a, )`
    parameters - All of these parameters are passed in as a tuple for the 
    third parameter in this order `(W_a, U_a, b_a, W_y, b_y)`

    Returns
    -------
    a_t
    y_t
    """
    W_a, U_a, b_a, W_y, b_y = parameters

    a_t = np.tanh(np.dot(x_t, W_a) + np.dot(a_prev, U_a) + b_a)
    y_t = softmax(np.dot(a_t, W_y) + b_y)
    return a_t, y_t

def rnn_forward(x, a_init, parameters):
    """
    Create a function to calculate the outputs and activations states for 
    each input of the sequence.

    Parameters
    x - A batch of inputs.  The inputs for the whole RNN layer are of shape 
    `(batch_size, sequence_length, n_x)`. That is to say, to compute the forward 
    pass, the layer is given number of samples as a batch. All samples in the batch 
    are sequences of length `sequence_length`. For example, for our weather forcasting 
    timeseries task, we had 5 days of samples taken once an hour, so the sequence length
    there was $5 \times 24 = 120$. And at a single measurement time there are `n_x` features 
    that were measured and are given as input, for example there were 14 weather measurements 
    in our weather prediction task.
    a_init - The initial value of the stat activations before iterating. Could be 0, but sometimes 
    we need to set the initial state (e.g. remember encoder/decoder architecture). Recall 
    the activations are of shape `(batch_size, n_a)`.
    parameters - All of the same parameters are again passed into this function, to be passed along
    and used by your `rnn_cell_forward()`.  They are passed in as a tuple for the third parameter 
    in this order `(W_a, U_a, b_a, W_y, b_y)`

    Returns
    -------
    a
    y
    """
    W_a, U_a, b_a, W_y, b_y = parameters
    batch_size, sequence_length, n_x = x.shape
    n_a = W_a.shape[1]
    n_y = W_y.shape[1]

    a = np.zeros((batch_size, sequence_length, n_a))
    y = np.zeros((batch_size, sequence_length, n_y))
    a_t = a_init
    
    for t in range(sequence_length):
        x_t = x[:, t, :]
        a_t, y_t = rnn_cell_forward(x_t, a_t, parameters)
        a[:, t, :] = a_t
        y[:, t, :] = y_t
    return a, y

def lstm_cell_forward(x_t, a_prev, m_prev, parameters):
    """
    Create a function with an iterative loop that will call the first 
    function the needed number of times to calculate the forward pass.

    Parmeters
    ---------
    x_t - 
    a_prev - 
    m_prev - 
    parameters - 

    Returns
    -------
    a_t
    m_t
    y_t
    """
    W_f, b_f, W_u, b_u, W_m, b_m, W_o, b_o, W_y, b_y = parameters
    C = np.concatenate((a_prev, x_t), axis=1)

    forget = sigmoid(np.dot(C, W_f) + b_f)
    update = sigmoid(np.dot(C, W_u) + b_u)
    memory = np.tanh(np.dot(C, W_m) + b_m)
    output = sigmoid(np.dot(C, W_o) + b_o)

    m_t = forget * m_prev + update * memory
    a_t = output * np.tanh(m_t)
    y_t = softmax(np.dot(a_t, W_y) + b_y)

    return a_t, m_t, y_t

def lstm_forward(x, a_init, parameters):
    """
    Create a function that will need to iterate over all `sequence_length` sequence items, 
    pulling out the batch of item for time `t`. This function returns the activation, 
    memory state and output predictions that result from all of the iterations. However 
    these will now be 3D tensors.

    Parameters
    ----------
    x - A batch of inputs.  The inputs for the whole RNN layer are of shape 
    `(batch_size, sequence_length, n_x)`. That is to say, to compute the forward 
    pass, the layer is given number of samples as a batch. All samples in the batch 
    are sequences of length `sequence_length`. For example, for our weather forcasting 
    timeseries task, we had 5 days of samples taken once an hour, so the sequence length
    there was $5 \times 24 = 120$. And at a single measurement time there are `n_x` features 
    that were measured and are given as input, for example there were 14 weather measurements 
    in our weather prediction task.
    a_init - The initial value of the stat activations before iterating. Could be 0, but sometimes 
    we need to set the initial state (e.g. remember encoder/decoder architecture). Recall 
    the activations are of shape `(batch_size, n_a)`.
    parameters - All of the same parameters are again passed into this function, to be passed along
    and used by your `rnn_cell_forward()`.  They are passed in as a tuple for the third parameter 
    in this order `(W_a, U_a, b_a, W_y, b_y)`

    Returns
    -------
    a
    m
    y
    """
    batch_size, sequence_length, n_x = x.shape
    n_a = a_init.shape[1]
    W_y = parameters[8]
    n_y = W_y.shape[1]

    a = np.zeros((batch_size, sequence_length, n_a))
    m = np.zeros((batch_size, sequence_length, n_a))
    y_hat = np.zeros((batch_size, sequence_length, n_y))

    a_t = a_init
    m_t = np.zeros_like(a_t)

    for t in range(sequence_length):
        x_t = x[:, t, :]
        a_t, m_t, y_t = lstm_cell_forward(x_t, a_t, m_t, parameters)

        a[:, t, :] = a_t
        m[:, t, :] = m_t
        y_hat[:, t, :] = y_t

    return a, m, y_hat

def softmax(x):
    """Calculates the softmax for each row of the input 2-D tensor matrix x.
    Your code should work for a row vector of shape (1, n) but also for general
    matrices of shape (m, n).  You shouldn't have to do anything special to handl
    a row vector, numpy broadcasting should work in either case.

    Arguments
    ---------
    x - A numpy matrix (2-D tensor) of shape (m, n)

    Returns
    -------
    s - A numpy matrix with the same shape (m, n) as the input argument x, with the computed softmax of x
    """
    # apply the exponential function element-wise to x
    x_exp = np.exp(x)

    # create a vector that holds the sum of each row
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # compute softmax(x) by dividing x_exp by the row sums.  It should automatically use numpy broadcasting
    s = x_exp / x_sum

    return s

def sigmoid(x):
    """Compute sigmoid of the input parameter x and return. In this version
    the input parameter might be a scalar, but it could be a list or
    a numpy array.  Your implementation should be vectorized and able to
    hanld all of these.

    Arguments
    ---------
    x - a scalar, python list or numpy array of real valued (float/double) numbers.

    Returns
    -------
    s - Result will be of the same shape as the input and will be the element wise
      calculation of the sigmoid for all values given as input.
    """
    s = 1 / (1 + np.exp(-x))
    return s