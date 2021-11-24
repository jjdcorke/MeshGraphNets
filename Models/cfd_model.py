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

"""Model for CylinderFlow."""

import tensorflow as tf
from tensorflow.keras.models import Model

import common
import core_model
import normalization


class CFDModel(Model):

    """Model for fluid simulation."""

    def __init__(self, model):
        super(CFDModel, self).__init__()
   
        self.model = model
        self._output_normalizer = normalization.Normalizer()
        self._node_normalizer = normalization.Normalizer()
        self._edge_normalizer = normalization.Normalizer()

    def call(self, graph, training = False):
        '''
        Pass a graph through the model

        :param graph: MultiGraph; the graph representing the raw mesh
        :param training: bool; if False, use inference mode
        :return: Tensor with shape (n, d), where n is the number of nodes and
                 d is the number of output dims; represents the node updates

        '''
        #normalize nodes and edge features
        new_node_features = self._node_normalizer(graph.node_features,training = training)
        new_edge_sets = [graph.edge_sets[0]._replace(features = self._edge_normalizer(graph.edge_sets[0].features, training = training))]
        graph = core_model.MultiGraph(new_node_features, new_edge_sets)
        
        #pass through encoder-processor-decoder architecture
        output = self.model(graph, training = training)
        
        return output

    def loss(self, graph, frame):
        """
        The loss function to use when training the model; the L2 distance
                between the ground-truth velocity and the model prediction
            :param graph: MultiGraph; the graph representing the raw mesh
            :param frame: dict; contains the ground-truth velocities
            :return: Tensor with shape (,) representing the loss value
        
        """
        network_output = self(graph, training=True)

        # build target velocity change
        cur_velocity = frame['velocity']
        target_velocity = frame['target|velocity']
        target_velocity_change = target_velocity - cur_velocity
        target_normalized = self._output_normalizer(target_velocity_change, training=True)

        # build loss
        loss_mask = tf.cast(tf.logical_or(tf.equal(frame['node_type'][:,0], common.NodeType.NORMAL),
                                          tf.equal(frame['node_type'][:,0], common.NodeType.OUTFLOW)), tf.float32)
        error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
        loss = tf.reduce_mean(error*loss_mask)
        return loss

    @tf.function(experimental_compile = True)
    def _update(self, graph, frame):
        """Integrate model outputs."""
        
        output = self(graph, training = False)
        
        velocity_update = self._output_normalizer.inverse(output)
        # integrate forward
        cur_velocity = frame['velocity']
        return cur_velocity + velocity_update

