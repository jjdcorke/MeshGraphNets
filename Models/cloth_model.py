"""
Implements a FlagSimple model
"""

import tensorflow as tf
from tensorflow.keras.models import Model

import common
import core_model
import normalization




class ClothModel(Model):
    """
    Model for static cloth simulation
    """
    @tf.function
    def __init__(self, model):
        """
        Create an instance of the cloth model
        :param model: EncodeProcessDecode; the core model to use
        """
        super(ClothModel, self).__init__()
        self.model = model

        # normalizer for the ground-truth acceleration
        self._output_normalizer = normalization.Normalizer()

        # normalizer for the raw node features before the encoder MLP
        self._node_normalizer = normalization.Normalizer()

        # normalizer for the raw edge features before the encoder MLP
        self._edge_normalizer = normalization.Normalizer()

    @tf.function
    def call(self, graph, training=False):
        """
        Pass a graph through the model
        :param graph: MultiGraph; the graph representing the raw mesh
        :param training: bool; if False, use inference mode
        :return: Tensor with shape (n, d), where n is the number of nodes and
                 d is the number of output dims; represents the node updates
        """
        # normalize node and edge features
        new_node_features = self._node_normalizer(graph.node_features, training=training)
        new_edge_sets = [graph.edge_sets[0]._replace(features=self._edge_normalizer(graph.edge_sets[0].features, training=training))]
        graph = core_model.MultiGraph(new_node_features, new_edge_sets)

        # pass through the encoder-processor-decoder architecture
        output = self.model(graph, training=training)

        return output
    @tf.function
    def loss(self, graph, frame):
        """
        The loss function to use when training the model; the L2 distance
            between the ground-truth acceleration and the model prediction
        :param graph: MultiGraph; the graph representing the raw mesh
        :param frame: dict; contains the ground-truth positions
        :return: Tensor with shape (,) representing the loss value
        """
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

    @tf.function
    def predict(self, graph, frame):
        """
        Predict the new position of the mesh after a single time-step
        :param graph: MultiGraph; the graph representing the raw mesh
        :param frame: dict; contains the current and previous positions of the mesh
        :return: Tensor with shape (n, d), where n is the number of nodes and
                 d is the number of dimensions in world-space.
        """
        output = self(graph, training=False)
        acceleration = self._output_normalizer.inverse(output)

        # integrate forward
        cur_position = frame['world_pos']
        prev_position = frame['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position
