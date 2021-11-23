"""
Implements an online normalization layer
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Normalizer(Layer):
    """
    Feature normalizer that accumulates statistics online
    """
    
    def __init__(self, max_accumulations=1000000, std_epsilon=1e-8, shift=True, scale=True):
        """
        Instantiate a normalization layer
        :param max_accumulations: int; the maximum number of updates to perform on the
                                  running variables
        :param std_epsilon: float; a small number in the std calculation to avoid
                            division by zero
        :param shift: bool; if False, don't shift the data to zero mean
        :param scale: bool; if False, don't scale the data to unit variance
        """
        super(Normalizer, self).__init__()
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon
        self.shift = shift
        self.scale = scale
    
    def build(self, input_shape):
        """
        Create the variables used by the layer. This method is called automatically
            by TensorFlow.
        :param input_shape: tuple of ints; the shape of the inputs to the layer
        :return: None
        """
        self._acc_count = tf.Variable(0, dtype=tf.float32, name='acc_count', trainable=False,
                                      aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
        self._num_accumulations = tf.Variable(0, dtype=tf.float32, name='num_acc', trainable=False,
                                              aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
        self._acc_sum = tf.Variable(tf.zeros(input_shape[-1], tf.float32), name='acc_sum', trainable=False,
                                    aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
        self._acc_sum_squared = tf.Variable(tf.zeros(input_shape[-1], tf.float32), name='acc_sum_squared', trainable=False,
                                            aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
    
    def call(self, x, training=False, mask=None):
        """
        Normalize the features of x independent of other samples, and add
            the new statistics to the running statistics
        :param x: Tensor; the tensor to normalize
        :param training: bool; if False, don't update the running statistics
        :return: a new Tensor with the same shape as x
        """
        if training and self._num_accumulations < self._max_accumulations:
            if mask is None:
                self._acc_count.assign_add(tf.cast(tf.shape(x)[0], tf.float32))
                self._acc_sum.assign_add(tf.reduce_sum(x, axis=0))
                self._acc_sum_squared.assign_add(tf.reduce_sum(x ** 2, axis=0))
                self._num_accumulations.assign_add(1.)
            else:
                self._acc_count.assign_add(tf.cast(tf.count_nonzero(mask), tf.float32))
                self._acc_sum.assign_add(tf.reduce_sum(x * mask, axis=0))
                self._acc_sum_squared.assign_add(tf.reduce_sum((x * mask) ** 2, axis=0))
                self._num_accumulations.assign_add(1.)

        return (x - self._mean()) / self._std_with_epsilon()
   
    def inverse(self, normalized_batch_data):
        """
        Inverse transformation of the normalizer
        :param normalized_batch_data: the Tensor to un-normalize
        :return: the new un-normalized Tensor with the same shape
        """
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _mean(self):
        """
        Get the running mean of inputs through this layer
        :return: Tensor with shape (d,) where d is the number of features
        """
        if self.shift:
            safe_count = tf.maximum(self._acc_count, 1.)
            return self._acc_sum / safe_count
        else:
            return 0

    def _std_with_epsilon(self):
        """
        Get the running standard deviation of inputs through this layer
        :return: Tensor with shape (d,) where d is the number of features
        """
        if self.scale:
            safe_count = tf.maximum(self._acc_count, 1.)
            std = tf.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
            return tf.math.maximum(std, self._std_epsilon)
        else:
            return 1
