import tensorflow as tf
from tensorflow import Tensor as ts
from tensorflow.python.ops import rnn
import my_rnn

def tile_repeat(n, repTime):
    '''
    create something like 111..122..2333..33 ..... n..nn 
    one particular number appears repTime consecutively 
    '''
    #print n, repTime
    idx = tf.range(n)
    idx = tf.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
    idx = tf.tile(idx, [1, repTime])  # Create multiple columns, each column has one number repeats repTime 
    y = tf.reshape(idx, [-1])
    return y

def gather_along_second_axis(x, idx):
    '''
    x has shape: [batch_size, sentence_length, word_dim]
    idx has shape: [batch_size, num_indices]
    Basically, in each batch, get words from sentence having index specified in idx
    However, since tensorflow does not fully support indexing,
    gather only work for the first axis. We have to reshape the input data, gather then reshape again
    '''
    idx1= tf.reshape(idx, [-1]) # [batch_size*num_indices]
    idx_flattened = tile_repeat(tf.shape(idx)[0], tf.shape(idx)[1]) * tf.shape(x)[1] + idx1
    y = tf.gather(tf.reshape(x, [-1,tf.shape(x)[2]]),  # flatten input
                idx_flattened)
    y = tf.reshape(y, tf.shape(x))
    return y


def gather_along_second_axis2(x, idx):
    '''
    x has shape: [batch_size, sentence_length, word_dim]
    idx has shape: [batch_size, sentence_length, num_indices]
    Basically, in each batch, get words from sentence having index specified in idx
    However, since tensorflow does not fully support indexing,
    gather only work for the first axis. We have to reshape the input data, gather then reshape again
    '''
    idx1= tf.reshape(idx, [-1]) # [batch_size * sentence_length * num_indices]
    idx_flattened = tile_repeat(tf.shape(idx)[0], tf.shape(idx)[1] * tf.shape(idx)[2]) * tf.shape(x)[1] + idx1
    y = tf.gather(tf.reshape(x, [-1,tf.shape(x)[2]]),  # flatten input
                idx_flattened)

    oshape = tf.shape(idx)
    y = tf.reshape(y, [oshape[0], oshape[1],oshape[2], -1])
    return y


