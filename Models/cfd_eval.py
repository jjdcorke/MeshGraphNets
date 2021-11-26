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
import os
from pathlib import Path
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import tensorflow as tf

import common
import core_model

import cfd_model

from dataset import load_dataset_eval



gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


model_dir = os.path.dirname(__file__)




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

def build_model(model, dataset):

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

    """Rolls out a model trajectory."""

    node_type = initial_frame['node_type']
    mask = tf.logical_or(tf.equal(node_type, common.NodeType.NORMAL),
                         tf.equal(node_type, common.NodeType.OUTFLOW))

    curr_velocity = initial_frame['velocity']
    curr_pressure = initial_frame['pressure']
    trajectory = []

    rollout_loop = tqdm(range(num_steps))
    for _ in rollout_loop:
        frame = {**initial_frame, 'velocity': curr_velocity}
        node_features, edge_features, senders, receivers, frame = frame_to_graph(frame)
        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

        next_velocity = model.predict(graph, frame)
        next_velocity = tf.where(mask, next_velocity, curr_velocity)
        trajectory.append(curr_velocity)

        curr_velocity = next_velocity

        frame = {**initial_frame, 'velocity': curr_velocity, 'pressure': curr_pressure}
        node_features, edge_features, senders, receivers, frame = frame_to_graph(frame)
        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

        next_velocity, next_pressure = model.predict(graph, frame)
        next_velocity = tf.where(mask, next_velocity, curr_velocity)
        next_pressure = tf.where(mask, next_pressure, curr_pressure)
        trajectory.append((curr_velocity, curr_pressure))
        curr_velocity, curr_pressure = next_velocity, next_pressure
    return tf.stack(trajectory)


def to_numpy(t):
    """
    If t is a Tensor, convert it to a NumPy array; otherwise do nothing
    """
    try:
        return t.numpy()
    except:
        return t


def avg_rmse():
    results_path = os.path.join(model_dir, 'results', 'cfd')
    results_prefixes = ['og_long-step4200000-loss0.01691.hdf5']

    for prefix in results_prefixes:
        all_errors = []
        for i in range(10):
            try:
                with open(os.path.join(results_path, prefix, f'{i:03d}.eval'), 'rb') as f:
                    data = pickle.load(f)
                    all_errors.append(data['errors'])
            except FileNotFoundError:
                continue

        keys = list(all_errors[0].keys())
        all_errors = {k: np.array([errors[k] for errors in all_errors]) for k in keys}

        for k, v in all_errors.items():
            print(prefix, k, np.mean(v))


def evaluate(model, inputs):
    """Performs model rollouts and create stats."""

    initial_state = {k: v[0] for k, v in inputs.items()}
    num_steps = inputs['cells'].shape[0]
    prediction = rollout(model, initial_state, num_steps)
    velocity_prediction = prediction[:][0]
    pressure_prediction = prediction[:][1]

    diff = tf.concat(velocity_prediction - inputs['velocity'], 
                     pressure_prediction - inputs['pressure'], -1)

    error = tf.reduce_mean(diff**2, axis=-1)
    scalars = {f'{horizon}_step_error': tf.math.sqrt(tf.reduce_mean(error[1:horizon + 1])).numpy()
               for horizon in [1, 10, 20, 50, 100, 200, 400, 598]}
    return scalars, (velocity_prediction, pressure_prediction) 

def run(checkpoint, data_path, num_trajectories):
    dataset = load_dataset_eval(
        path=data_path,
        split='test',
        fields=['velocity', 'pressure'],
        add_history=False
    )
    model = core_model.EncodeProcessDecode(
        output_dims=2,
        embed_dims=128,
        num_layers=3,
        num_iterations=15,
        num_edge_types=1
    )
    model = cfd_model.CFDModel(model)
    build_model(model, dataset)
    model.load_weights(checkpoint, by_name=True)

    Path(os.path.join(model_dir, 'results', 'cfd', os.path.split(checkpoint)[-1])).mkdir(exist_ok=True, parents=True)
    for i, trajectory in enumerate(dataset.take(num_trajectories)):
        rmse_error, predicted_trajectory = evaluate(model, trajectory)
        print(f'RMSE Errors: {rmse_error}')

        data = {**trajectory, 'true_velocity': trajectory['velocity'],
                              'pred_velocity': predicted_trajectory[0],
                              'true_pressure': trajectory['pressure'],
                              'pred_pressure': predicted_trajectory[1],
                              'errors': rmse_error}
        data = {k: to_numpy(v) for k, v in data.items()}
        save_path = os.path.join(model_dir, 'results', 'cfd', os.path.split(checkpoint)[-1], f'{i:03d}.eval')
        with open(save_path, "wb") as fp:
            pickle.dump(data, fp)
            print(f'Evaluation results saved in {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("data_path", help="Path to dataset")
    parser.add_argument("num_trajectories", type=int, help="Number of trajectories to evaluate")
    args = parser.parse_args()
    run(args.checkpoint, args.data_path, args.num_trajectories)


if __name__ == "__main__":
    main()
    # avg_rmse()
