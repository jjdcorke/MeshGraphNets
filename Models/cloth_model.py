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

    def create_graph(self, frame, world_pos, prev_world_pos, senders, receivers):
        # TODO: support for wind velocity

        velocity = world_pos - prev_world_pos
        node_type = tf.one_hot(frame['node_type'][:, 0], common.NodeType.SIZE)

        node_features = tf.concat([velocity, node_type], axis=-1)

        # construct graph edges
        relative_world_pos = tf.gather(world_pos, senders) - tf.gather(world_pos, receivers)
        relative_mesh_pos = tf.gather(frame['mesh_pos'], senders) - tf.gather(frame['mesh_pos'], receivers)
        edge_features = tf.concat([
            relative_world_pos,
            tf.norm(relative_world_pos, axis=-1, keepdims=True),
            relative_mesh_pos,
            tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])
        return graph

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

    def loss(self, frame, senders, receivers):
        """
        The loss function to use when training the model; the L2 distance
            between the ground-truth acceleration and the model prediction
        :param graph: MultiGraph; the graph representing the raw mesh
        :param frame: dict; contains the ground-truth positions
        :return: Tensor with shape (,) representing the loss value
        """
        # build first model output
        graph1 = self.create_graph(frame, frame['world_pos'], frame['prev|world_pos'], senders, receivers)
        output1 = self(graph1, training=True)

        # build target1 acceleration
        target1_acceleration = frame['target1|world_pos'] - 2 * frame['world_pos'] + frame['prev|world_pos']
        target1_normalized = self._output_normalizer(target1_acceleration, training=True)

        # build predicted world_pos 1
        pred1 = 2 * frame['world_pos'] + self._output_normalizer.inverse(output1) - frame['prev|world_pos']

        # build second model output
        graph2 = self.create_graph(frame, pred1, frame['world_pos'], senders, receivers)
        output2 = self(graph2, training=True)

        # build target2 acceleration
        target2_acceleration = frame['target2|world_pos'] - 2 * frame['target1|world_pos'] + frame['world_pos']
        target2_normalized = self._output_normalizer(target2_acceleration, training=True)

        # build predicted world_pos 2
        pred2 = 2 * pred1 + self._output_normalizer.inverse(output2) - frame['world_pos']

        # build third model output
        graph3 = self.create_graph(frame, pred2, pred1, senders, receivers)
        output3 = self(graph3, training=True)

        # build target3 acceleration
        target3_acceleration = frame['target3|world_pos'] - 2 * frame['target2|world_pos'] + frame['target1|world_pos']
        target3_normalized = self._output_normalizer(target3_acceleration, training=True)

        # build loss
        loss_mask = tf.cast(tf.equal(frame['node_type'][:, 0], common.NodeType.NORMAL), tf.float32)
        error1 = tf.reduce_sum(tf.math.square(target1_normalized - output1), axis=1)
        error2 = tf.reduce_sum(tf.math.square(target2_normalized - output2), axis=1)
        error3 = tf.reduce_sum(tf.math.square(target3_normalized - output3), axis=1)
        loss = tf.reduce_mean((error1 + error2 + error3) / 3 * loss_mask)

        return loss

    @tf.function(experimental_compile=True)
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
