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
        # with self._enter_variable_scope():
        self.model = model
        self._output_normalizer = normalization_tf2.Normalizer(size=3)
        self._node_normalizer = normalization_tf2.Normalizer(size=3 + common.NodeType.SIZE)
        self._edge_normalizer = normalization_tf2.Normalizer(size=7)

    def _build_graph(self, inputs, training=False):
        """Builds input graph."""
        # construct graph nodes
        velocity = inputs['world_pos'] - inputs['prev|world_pos']
        node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
        node_features = tf.concat([velocity, node_type], axis=-1)

        # construct graph edges
        senders, receivers = common.triangles_to_edges(inputs['cells'])
        relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                              tf.gather(inputs['world_pos'], receivers))
        relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                             tf.gather(inputs['mesh_pos'], receivers))
        edge_features = tf.concat([
            relative_world_pos,
            tf.norm(relative_world_pos, axis=-1, keepdims=True),
            relative_mesh_pos,
            tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

        mesh_edges = core_model_tf2.EdgeSet(
            features=self._edge_normalizer(edge_features, training=training),
            receivers=receivers,
            senders=senders)

        return core_model_tf2.MultiGraph(self._node_normalizer(node_features, training=training), edge_sets=[mesh_edges])

    def call(self, inputs, training=False):
        graph = self._build_graph(inputs, training=training)
        output = self.model(graph, training=training)
        return output

    def loss(self, inputs):
        """L2 loss on position."""
        network_output = self(inputs, training=True)

        # build target acceleration
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        target_position = inputs['target|world_pos']
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self._output_normalizer(target_acceleration, training=True)

        # build loss
        loss_mask = tf.cast(tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL), tf.float32)
        error = tf.reduce_sum(tf.math.square(target_normalized - network_output), axis=1)
        loss = tf.reduce_mean(error * loss_mask)
        return loss

    def _update(self, inputs, output):
        """Integrate model outputs."""
        acceleration = self._output_normalizer.inverse(output)

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position
