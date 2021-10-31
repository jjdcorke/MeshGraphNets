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
"""Core learned graph net model."""

import collections
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LayerNormalization


EdgeSet = collections.namedtuple('EdgeSet', ['features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class MLP(Model):
    def __init__(self, hidden_dims, layer_norm=True):
        super(MLP, self).__init__()

        self.dense_layers = []
        for i, dim in enumerate(hidden_dims[::-1]):
            activation = None if i == 0 else 'relu'
            self.dense_layers.insert(0, Dense(dim, activation=activation))

        if layer_norm:
            self.dense_layers.append(LayerNormalization())

    def call(self, x, training=False):
        for layer in self.dense_layers:
            x = layer(x)
        return x


class GraphNetBlock(Model):

    def __init__(self, embed_dims, num_layers, num_edge_types=2):
        super(GraphNetBlock, self).__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types

        self.node_update = MLP([embed_dims] * num_layers)
        self.edge_updates = [MLP([embed_dims] * num_layers) for _ in range(num_edge_types)]

    def _update_edge_features(self, node_features, edge_set, index):
        sender_features = tf.gather(node_features, edge_set.senders)
        receiver_features = tf.gather(node_features, edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        return self.edge_updates[index](tf.concat(features, axis=-1))

    def _update_node_features(self, node_features, edge_sets):
        num_nodes = tf.shape(node_features)[0]
        features = [node_features]
        for edge_set in edge_sets:
            features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                         edge_set.receivers,
                                                         num_nodes))
        return self.node_update(tf.concat(features, axis=-1))

    def call(self, graph, training=False):

        # update edge features
        new_edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            updated_features = self._update_edge_features(graph.node_features, edge_set, i)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # update node features
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features = new_node_features + graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]

        return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(Model):
    def __init__(self, output_dims, embed_dims, num_layers, num_iterations, num_edge_types=2):
        super(EncodeProcessDecode, self).__init__()
        self.output_dims = output_dims
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_iterations = num_iterations
        self.num_edge_types = num_edge_types

        # encoder MLPs
        self.node_encoder = MLP([embed_dims] * num_layers)
        self.edge_encoders = [MLP([embed_dims] * num_layers) for _ in range(num_edge_types)]

        # graph message-passing blocks
        self.mp_blocks = []
        for _ in range(num_iterations):
            self.mp_blocks.append(GraphNetBlock(embed_dims, num_layers, num_edge_types=num_edge_types))

        # decoder MLPs
        self.node_decoder = MLP([embed_dims] * (num_layers - 1) + [output_dims])

    def _encoder(self, graph):
        node_features = self.node_encoder(graph.node_features)
        edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            edge_features = self.edge_encoders[i](edge_set.features)
            edge_sets.append(edge_set._replace(features=edge_features))

        return MultiGraph(node_features, edge_sets)

    def _decoder(self, graph):
        return self.node_decoder(graph.node_features)

    def call(self, graph, training=False):
        graph = self._encoder(graph)
        for i in range(self.num_iterations):
            graph = self.mp_blocks[i](graph)
        return self._decoder(graph)
