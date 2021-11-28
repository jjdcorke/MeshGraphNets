"""
Implements the core model common to MeshGraphNets
"""

import collections
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LayerNormalization


EdgeSet = collections.namedtuple('EdgeSet', ['features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class MLP(Model):
    """
    A simple feed-forward neural network with ReLU activation
    """
    def __init__(self, hidden_dims, layer_norm=True):
        """
        Create an MLP
        :param hidden_dims: list of ints representing the size of the hidden layers.
                            For example, hidden_dims=[128, 10] would create a two-layer
                            MLP with 128 hidden units and 10 output units.
        :param layer_norm: bool; if True, perform layer normalization after the last layer
        """
        super(MLP, self).__init__()

        self.dense_layers = []
        for i, dim in enumerate(hidden_dims[::-1]):
            activation = None if i == 0 else 'relu'
            self.dense_layers.insert(0, Dense(dim, activation=activation))

        if layer_norm:
            self.dense_layers.append(LayerNormalization())

    def call(self, x, training=False):
        """
        Pass the input x through the MLP
        :param x: Tensor with shape (n, d), where n is the batch size and d is the
                  number of input dimensions
        :param training: unused
        :return: Tensor with shape (n, d2), where n is the batch size and d2 is the
                 number of output dimensions
        """
        for layer in self.dense_layers:
            x = layer(x)
        return x


class GraphNetBlock(Model):
    """
    A message-passing block
    """

    def __init__(self, embed_dims, num_layers, num_edge_types=2):
        """
        Create a message-passing block
        :param embed_dims: int; size of the hidden layers and the latent space
        :param num_layers: int; number of layers in the MLPs
        :param num_edge_types: int; number of edge types
        """
        super(GraphNetBlock, self).__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types

        # MLP for the node update rule
        self.node_update = MLP([embed_dims] * num_layers)

        # separate MLPs for the edge update rules
        self.edge_updates = [MLP([embed_dims] * num_layers) for _ in range(num_edge_types)]

    def _update_edge_features(self, node_features, edge_set, index):
        """
        Get the new edge features by aggregating the features of the
            two adjacent nodes
        :param node_features: Tensor with shape (n, d), where n is the number of nodes
                              and d is the number of input dims
        :param edge_set: EdgeSet; the edge set to update
        :param index: int; the index of the edge set in MultiGraph.edge_sets
        :return: Tensor with shape (m, d2), where m is the number of edges and
                 d2 is the number of output dims
        """
        sender_features = tf.gather(node_features, edge_set.senders)
        receiver_features = tf.gather(node_features, edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        return self.edge_updates[index](tf.concat(features, axis=-1))


    def _update_node_features(self, node_features, edge_sets):
        """
        Get the new node features by aggregating the features of all
            adjacent edges
        :param node_features: Tensor with shape (n, d), where n is the number of nodes
                              and d is the number of input dims
        :param edge_sets: list of EdgeSets; all edge sets in the graph
        :return: Tensor with shape (n, d2), where n is the number of nodes and d2
                 is the number of output dims
        """
        num_nodes = tf.shape(node_features)[0]
        features = [node_features]
        for edge_set in edge_sets:
            # perform sum aggregation
            features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                         edge_set.receivers,
                                                         num_nodes))
        return self.node_update(tf.concat(features, axis=-1))

    def call(self, graph, training=False):
        """
        Perform the message-passing on the graph
        :param graph: MultiGraph; the input graph
        :param training: unused
        :return: MultiGraph; the resulting graph after updating features
        """
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
    """
    The composite 3-part architecture consisting of an encoder, a processor,
        and a decoder
    """

    def __init__(self, output_dims, embed_dims, num_layers, num_iterations, num_edge_types=2):
        """
        Create a model
        :param output_dims: int; number of output dimensions
        :param embed_dims: int; size of all hidden layers
        :param num_layers: int; number of layers in all MLPs
        :param num_iterations: int; number of message-passing iterations
        :param num_edge_types: int; number of edge types
        """
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
        self.node_decoder = MLP([embed_dims] * (num_layers - 1) + [output_dims], layer_norm=False)

    def _encoder(self, graph):
        """
        Map the input features onto a latent space for processing
        :param graph: MultiGraph; the input graph representing the raw mesh features
        :return: MultiGraph; the new latent graph
        """
        node_features = self.node_encoder(graph.node_features)
        edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            edge_features = self.edge_encoders[i](edge_set.features)
            edge_sets.append(edge_set._replace(features=edge_features))

        return MultiGraph(node_features, edge_sets)

    def _decoder(self, graph, wind_velocities=None):
        """
        Readout of the final node features
        :param graph: MultiGraph; the graph after all message-passing iterations
        :return: Tensor with shape (n, d), where n is the number of nodes and d
                 is the number of output dims; represents the node update
        """
        if wind_velocities is not None:
            decoder_input = tf.concat([graph.node_features, wind_velocities], axis=1)
            return self.node_decoder(decoder_input)
        else:
            return self.node_decoder(graph.node_features)

    def call(self, graph, wind_velocities=None, training=False):
        """
        Pass a graph through the model
        :param graph: MultiGraph; represents the mesh with raw node and edge features
        :param training: unused
        :return: Tensor with shape (n, d), where n is the number of nodes and d
                 is the number of output dims; represents the node update
        """
        graph = self._encoder(graph)
        for i in range(self.num_iterations):
            graph = self.mp_blocks[i](graph)
        return self._decoder(graph, wind_velocities)
