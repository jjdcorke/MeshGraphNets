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
import argparse

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import common
import core_model
import cfd_model
from dataset import load_dataset_train

import datetime


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


def frame_to_graph(frame):
    """Builds input graph."""

    # construct graph nodes
    node_type = tf.one_hot(frame['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([frame['velocity'], frame['pressure'], node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(frame['cells'])
    relative_mesh_pos = (tf.gather(frame['mesh_pos'], senders) -
                         tf.gather(frame['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    del frame['cells']

    return node_features, edge_features, senders, receivers, frame


def build_model(model, optimizer, dataset, checkpoint=None):
    """Initialize the model"""
    node_features, edge_features, senders, receivers, frame = next(iter(dataset))
    graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

    # call the model once to process all input shapes
    model.loss(graph, frame)

    # get the number of trainable parameters
    total = 0
    for var in model.trainable_weights:
        total += np.prod(var.shape)
    print(f'Total trainable parameters: {total}')

    if checkpoint:
        opt_weights = np.load(f'{checkpoint}_optimizer.npy', allow_pickle=True)

        dummy_grads = [tf.zeros_like(w) for w in model.trainable_weights]
        optimizer.apply_gradients(zip(dummy_grads, model.trainable_weights))

        # only now set the weights of the optimizer and model
        optimizer.set_weights(opt_weights)
        model.load_weights(checkpoint, by_name=True)


def train(data_path=os.path.join(os.path.dirname(__file__), 'data', 'cylinder_flow'),num_steps=10000000, checkpoint = None):
    dataset = load_dataset_train(
        path=data_path,
        split='train',
        fields=['velocity', 'pressure'],
        add_history=False,
        noise_scale=0.02,
        noise_gamma=1.0
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
    model = cfd_model.CFDModel(model)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=num_steps // 2, decay_rate=0.1)
    optimizer = Adam(learning_rate=lr)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'Visualization/logs/train/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # build the model
    build_model(model, optimizer, dataset, checkpoint = checkpoint)

    #@tf.function(jit_compile=True)
    @tf.function(experimental_relax_shapes=True)
    def warmup(graph, frame):
        loss = model.loss(graph, frame)
        return loss

    #@tf.function(jit_compile=True)
    @tf.function(experimental_relax_shapes=True)
    def train_step(graph, frame):
        with tf.GradientTape() as tape:
            loss = model.loss(graph, frame)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss

    dataset_iter = iter(dataset)
    train_loop = tqdm(range(num_steps))
    moving_loss = 0
    for s in train_loop:
        node_features, edge_features, senders, receivers, frame = next(dataset_iter)
        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

        if s < 1000:
            loss = warmup(graph, frame)
        else:
            loss = train_step(graph, frame)

        moving_loss = 0.98 * moving_loss + 0.02 * loss

        if s%500 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss',loss,step = s) #s for training session

        train_loop.set_description(f'Step {s}/{num_steps}, Loss {moving_loss:.5f}')

        if s != 0 and s % 50000 == 0:
            filename = f'weights-step{s:07d}-loss{moving_loss:.5f}.hdf5'
            model.save_weights(os.path.join(os.path.dirname(__file__), 'checkpoints_cfd_long', filename))
            np.save(os.path.join(os.path.dirname(__file__), 'checkpoints_cfd_long', f'{filename}_optimizer.npy'), optimizer.get_weights())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", help="Path to checkpoint file used to resume training")
    parser.add_argument("data_path", help="Path to dataset")
    parser.add_argument("num_steps", type=int, help="Number of itterations to train (default :1e6)")
    args = parser.parse_args()

    train(args.data_path, num_steps=args.num_steps, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
