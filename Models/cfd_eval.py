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
    node_features = tf.concat([frame['velocity'], node_type], axis=-1)

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


    node_type = initial_frame['node_type'][:, 0]
    mask = tf.logical_or(tf.equal(node_type, common.NodeType.NORMAL),
                       tf.equal(node_type, common.NodeType.OUTFLOW))
     
     # someone continue where I Left off

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
        return t.numpy()
    except:
        return t   
    



#Deepmind part to change

def evaluate(model, inputs):

    
    """Performs model rollouts and create stats."""

    
    initial_state = {k: v[0] for k, v in inputs.items()}
    num_steps = inputs['cells'].shape[0]
    prediction = rollout(model, initial_state, num_steps)

    error = tf.reduce_mean((prediction - inputs['velocity'])**2, axis=-1)
    scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200]}
    traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_velocity': inputs['velocity'],
      'pred_velocity': prediction
  }

    return scalars, traj_ops

def run(checkpoint, data_path, num_trajectories):
    dataset = load_dataset_eval(
        path=data_path,
        split='test',
        fields=['world_pos'],
        add_history=True
    )
    model = core_model.EncodeProcessDecode(
        output_dims=6,
        embed_dims=128,
        num_layers=2,
        num_iterations=15,
        num_edge_types=1
    )
    model = cfd_model.CFDModel(model)
    build_model(model, dataset)
    model.load_weights(checkpoint, by_name=True)
    for i, trajectory in enumerate(dataset.take(num_trajectories)):
        rmse_error, predicted_trajectory = evaluate(model, dataset)
        data = {**trajectory, 'true_world_pos': trajectory['world_pos'], 'pred_world_pos': predicted_trajectory, 'errors': rmse_error}
        data = {k: to_numpy(v) for k, v in data.items()}
        save_path = os.path.join(model_dir, 'results', os.path.split(checkpoint)[-1], f'{i:03d}.eval')
        with open(save_path, "wb") as fp:
            pickle.dump(data, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("data_path", help="Path to dataset")
    parser.add_argument("num_trajectories", type=int, help="Number of trajectories to evaluate")
    args = parser.parse_args()
    run(args.checkpoint, args.data_path, args.num_trajectories)


if __name__ == "__main__":
    main()
