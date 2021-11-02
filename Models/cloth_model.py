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
"""Model for FlagSimple."""

import tensorflow as tf
from tensorflow.keras.models import Model

import common
import core_model
import normalization


class ClothModel(Model):
    """Model for static cloth simulation."""
    def __init__(self, model):
        super(ClothModel, self).__init__()

        self.model = model
        self._output_normalizer = normalization.Normalizer()
        self._node_normalizer = normalization.Normalizer()
        self._edge_normalizer = normalization.Normalizer()

    def call(self, graph, training=False):
        # normalize node and edge features
        new_node_features = self._node_normalizer(graph.node_features, training=training)
        new_edge_sets = [graph.edge_sets[0]._replace(features=self._edge_normalizer(graph.edge_sets[0].features, training=training))]
        graph = core_model.MultiGraph(new_node_features, new_edge_sets)

        output = self.model(graph, training=training)
        return output

    def loss(self, graph, frame):
        """L2 loss on position."""
        network_output = self(graph, training=True)

        # build target acceleration
        cur_position = frame['world_pos']
        prev_position = frame['prev|world_pos']
        target_position = frame['target|world_pos']
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self._output_normalizer(target_acceleration, training=True)

        # build loss
        loss_mask = tf.cast(tf.equal(frame['node_type'][:, 0], common.NodeType.NORMAL), tf.float32)
        error = tf.reduce_sum(tf.math.square(target_normalized - network_output), axis=1)
        loss = tf.reduce_mean(error * loss_mask)
        return loss

    def predict(self, graph, frame):
        """Integrate model outputs."""
        output = self(graph, training=False)
        acceleration = self._output_normalizer.inverse(output)

        # integrate forward
        cur_position = frame['world_pos']
        prev_position = frame['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position
