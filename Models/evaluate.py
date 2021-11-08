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
from pathlib import Path

from tqdm import tqdm
import numpy as np
import tensorflow as tf

import common
import core_model
import cloth_model
from dataset import load_dataset_eval


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

    del frame['cells']

    return node_features, edge_features, senders, receivers, frame


def build_model(model, dataset):
    """Initialize the model"""
    traj = next(iter(dataset))
    frame = {k: v[0] for k, v in traj.items()}
    node_features, edge_features, senders, receivers, frame = frame_to_graph(frame)
    graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

    # call the model once to process all input shapes
    model.loss(graph, frame)

    # get the number of trainable parameters
    total = 0
    for var in model.trainable_weights:
        total += np.prod(var.shape)
    print(f'Total trainable parameters: {total}')


def rollout(model, initial_frame, num_steps):
    """Rollout a model trajectory."""
    mask = tf.equal(initial_frame['node_type'], common.NodeType.NORMAL)

    prev_pos = initial_frame['prev|world_pos']
    curr_pos = initial_frame['world_pos']
    trajectory = []

    rollout_loop = tqdm(range(num_steps))
    for _ in rollout_loop:
        frame = {**initial_frame, 'prev|world_pos': prev_pos, 'world_pos': curr_pos}
        node_features, edge_features, senders, receivers, frame = frame_to_graph(frame)
        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

        next_pos = model.predict(graph, frame)
        next_pos = tf.where(mask, next_pos, curr_pos)
        trajectory.append(curr_pos)

        prev_pos, curr_pos = curr_pos, next_pos

    return tf.stack(trajectory)


def to_numpy(t):
    """
    If t is a Tensor, convert it to a NumPy array; otherwise do nothing
    """
    try:
        return t.to_numpy()
    except:
        return t


def evaluate(checkpoint_file, num_trajectories):
    dataset = load_dataset_eval(
        path='data/flag_simple',
        split='test',
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

    # build the model
    build_model(model, dataset)

    model.load_weights(checkpoint_file, by_name=True)

    preds = []
    for trajectory in dataset.take(num_trajectories):
        initial_frame = {k: v[0] for k, v in trajectory.items()}
        predicted_trajectory = rollout(model, initial_frame, trajectory['cells'].shape[0])

        error = tf.reduce_mean(tf.square(predicted_trajectory - trajectory['world_pos']), axis=-1)
        rmse_errors = {f'{horizon}_step_error': tf.math.sqrt(tf.reduce_mean(error[1:horizon + 1])).numpy()
                       for horizon in [1, 10, 20, 50, 100, 200, 398]}
        print(f'RMSE Errors: {rmse_errors}')

        preds.append({**trajectory, 'world_pos': predicted_trajectory, 'errors': rmse_errors})

    preds = [{k: to_numpy(v) for k, v in pred.items()} for pred in preds]

    Path('results').mkdir(exist_ok=True)
    save_path = os.path.join('results', f'{os.path.split(checkpoint_file)[-1]}.eval')
    with open(save_path, 'wb') as f:
        pickle.dump(preds, f)
        print(f'Evaluation results saved in {save_path}')

    return preds


def main():
    evaluate('checkpoints/weights-step2100000-loss0.0680.hdf5', 10)


if __name__ == '__main__':
    main()