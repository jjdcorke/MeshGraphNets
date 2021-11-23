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
from dataset import load_dataset_train, load_dataset_eval
from evaluate import rollout

import datetime


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

def add_noise(field, scale):
    noise = tf.random.normal(tf.shape(field), stddev=scale, dtype=tf.float32)
    field += noise
    return field

def frame_to_graph(frame, wind=False):
    """Builds input graph."""

    # construct graph nodes
    velocity = frame['world_pos'] - frame['prev|world_pos']
    node_type = tf.one_hot(frame['node_type'][:, 0], common.NodeType.SIZE)

    node_features = tf.concat([velocity, node_type], axis=-1)
    if wind:
        wind_velocities = tf.ones([len(velocity), len(frame['wind_velocity'])]) * frame['wind_velocity']
        wind_velocities = add_noise(wind_velocities, scale=0.01)
        node_features = tf.concat([node_features, wind_velocities], axis=-1)

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


def validation(model, dataset, num_trajectories=5):
    print('\nEvaluating...')
    horizons = [1, 10, 20, 50, 100, 200, 398]
    all_errors = {horizon: [] for horizon in horizons}
    for i, trajectory in enumerate(dataset.take(num_trajectories)):
        initial_frame = {k: v[0] for k, v in trajectory.items()}
        predicted_trajectory = rollout(model, initial_frame, trajectory['cells'].shape[0])

        error = tf.reduce_mean(tf.square(predicted_trajectory - trajectory['world_pos']), axis=-1)
        for horizon in horizons:
            all_errors[horizon].append(tf.math.sqrt(tf.reduce_mean(error[1:horizon + 1])).numpy())

    return {k: np.mean(v) for k, v in all_errors.items()}


def train(num_steps=10000000, checkpoint=None, wind=False):
    dataset = load_dataset_train(
        path=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
        split='train',
        fields=['world_pos'],
        add_history=True,
        noise_scale=0.003,
        noise_gamma=0.1
    )
    dataset = dataset.map(lambda frame: frame_to_graph(frame, wind=wind), num_parallel_calls=8)
    dataset = dataset.prefetch(16)

    valid_dataset = load_dataset_eval(
        path=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
        split='valid',
        fields=['world_pos'],
        add_history=True
    )

    model = core_model.EncodeProcessDecode(
        output_dims=3,
        embed_dims=128,
        num_layers=3,
        num_iterations=15,
        num_edge_types=1
    )
    model = cloth_model.ClothModel(model)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=num_steps // 2, decay_rate=0.1)
    optimizer = Adam(learning_rate=lr)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'Visualization/logs/train/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # build the model
    build_model(model, optimizer, dataset, checkpoint = checkpoint)
    # build_model(model, optimizer, dataset, checkpoint='checkpoints/weights-step2700000-loss0.0581.hdf5')

    @tf.function(experimental_compile=True)
    def warmup(graph, frame):
        loss = model.loss(graph, frame)
        return loss

    @tf.function(experimental_compile=True)
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
            model.save_weights(os.path.join(os.path.dirname(__file__), 'checkpoints_og_noise', filename))
            np.save(os.path.join(os.path.dirname(__file__), 'checkpoints_og_noise', f'{filename}_optimizer.npy'), optimizer.get_weights())

            # perform validation
            errors = validation(model, valid_dataset)
            print(', '.join([f'{k}-step RMSE: {v}' for k, v in errors.items()]))


def main():

    train()


if __name__ == '__main__':
    main()
