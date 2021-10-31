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
"""Runs the learner/evaluator."""

import os
import pickle

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import common
import core_model
import cloth_model
from dataset import load_dataset


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


def frame_to_graph(frame):
    """Builds input graph."""

    # construct graph nodes
    velocity = frame['world_pos'] - frame['prev|world_pos']
    node_type = tf.one_hot(frame['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(frame['cells'])
    relative_world_pos = (tf.gather(frame['world_pos'], senders) -
                          tf.gather(frame['world_pos'], receivers))
    relative_mesh_pos = (tf.gather(frame['mesh_pos'], senders) -
                         tf.gather(frame['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)
    # mesh_edges = core_model.EdgeSet(edge_features, senders, receivers)

    del frame['cells']

    # return core_model.MultiGraph(node_features, edge_sets=[mesh_edges]), frame
    return node_features, edge_features, senders, receivers, frame


def train():
    dataset = load_dataset(
        path='data/flag_simple',
        split='train',
        fields=['world_pos'],
        add_history=True,
        noise_scale=0.003,
        noise_gamma=0.1
    )
    dataset = dataset.map(frame_to_graph, num_parallel_calls=8)
    dataset = dataset.prefetch(16)

    model = core_model.EncodeProcessDecode(
        output_dims=3,
        embed_dims=128,
        num_layers=3,
        num_iterations=15,
        num_edge_types=1
    )
    model = cloth_model.ClothModel(model)
    optimizer = Adam(learning_rate=1e-4)

    @tf.function(jit_compile=True)
    def warmup(graph, frame):
        loss = model.loss(graph, frame)
        return loss

    @tf.function(jit_compile=True)
    def train_step(graph, frame):
        with tf.GradientTape() as tape:
            loss = model.loss(graph, frame)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss

    num_steps = 1000000
    dataset_iter = iter(dataset)
    train_loop = tqdm(range(num_steps))
    moving_loss = 0
    for s in train_loop:
        node_features, edge_features, senders, receivers, frame = next(dataset_iter)
        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

        if s == 50:
            tf.profiler.experimental.start('logdir')
        elif s == 55:
            tf.profiler.experimental.stop()

        if s < 1000:
            loss = warmup(graph, frame)
        else:
            loss = train_step(graph, frame)

        moving_loss = 0.98 * moving_loss + 0.02 * loss

        train_loop.set_description(f'Step {s}/{num_steps}, Loss {moving_loss:.2f}')

        if s != 0 and s % 50000 == 0:
            model.save_weights(os.path.join(os.path.dirname(__file__), 'checkpoints', f'weights-step{s:07d}-loss{moving_loss:.4f}.hdf5'))


def main():
    train()


if __name__ == '__main__':
    main()
