# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Online data normalization."""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Normalizer(Layer):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, max_accumulations=1000000, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon

    def build(self, input_shape):
        self._acc_count = tf.Variable(0, dtype=tf.float32, name='acc_count', trainable=False,
                                      aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
        self._num_accumulations = tf.Variable(0, dtype=tf.float32, name='num_acc', trainable=False,
                                              aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
        self._acc_sum = tf.Variable(tf.zeros(input_shape[-1], tf.float32), name='acc_sum', trainable=False,
                                    aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)
        self._acc_sum_squared = tf.Variable(tf.zeros(input_shape[-1], tf.float32), name='acc_sum_squared', trainable=False,
                                            aggregation=tf.VariableAggregation.SUM, synchronization=tf.VariableSynchronization.ON_READ)

    def call(self, x, training=False):
        if training and self._num_accumulations < self._max_accumulations:
            self._acc_count.assign_add(tf.cast(tf.shape(x)[0], tf.float32))
            self._acc_sum.assign_add(tf.reduce_sum(x, axis=0))
            self._acc_sum_squared.assign_add(tf.reduce_sum(x ** 2, axis=0))
            self._num_accumulations.assign_add(1.)

        return (x - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _mean(self):
        safe_count = tf.maximum(self._acc_count, 1.)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = tf.maximum(self._acc_count, 1.)
        std = tf.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return tf.math.maximum(std, self._std_epsilon)
