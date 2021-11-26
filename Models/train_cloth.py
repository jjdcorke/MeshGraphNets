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


def frame_to_senders_and_receivers(frame):
    senders, receivers = common.triangles_to_edges(frame['cells'])
    del frame['cells']
    return frame, senders, receivers


def build_model(model, optimizer, dataset, checkpoint=None):
    """Initialize the model"""
    frame, senders, receivers = next(iter(dataset))

    # call the model once to process all input shapes
    model.loss(frame, senders, receivers)

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


def train(num_steps=10000000, checkpoint=None):
    dataset = load_dataset_train(
        path=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
        split='train',
        fields=['world_pos']
    )
    dataset = dataset.map(lambda frame: frame_to_senders_and_receivers(frame), num_parallel_calls=8)
    dataset = dataset.prefetch(16)

    valid_dataset = load_dataset_eval(
        path=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
        split='valid',
        fields=['world_pos']
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
    trainlossdir = train_log_dir+'/loss'
    vallossdir = train_log_dir+'/validation'

    loss_summary_writer = tf.summary.create_file_writer(trainlossdir)
    val_summary_writer = tf.summary.create_file_writer(vallossdir)

    # build the model
    build_model(model, optimizer, dataset, checkpoint = checkpoint)
    # build_model(model, optimizer, dataset, checkpoint='checkpoints/weights-step2700000-loss0.0581.hdf5')

    @tf.function(experimental_compile=True)
    def warmup(frame, senders, receivers):
        loss = model.loss(frame, senders, receivers)
        return loss

    @tf.function(experimental_compile=True)
    def train_step(frame, senders, receivers):
        with tf.GradientTape() as tape:
            loss = model.loss(frame, senders, receivers)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss

    dataset_iter = iter(dataset)
    train_loop = tqdm(range(num_steps))
    moving_loss = 0
    for s in train_loop:
        frame, senders, receivers = next(dataset_iter)

        if s < 1000:
            loss = warmup(frame, senders, receivers)
        else:
            loss = train_step(frame, senders, receivers)

        moving_loss = 0.98 * moving_loss + 0.02 * loss

        if s%500 == 0:
            with loss_summary_writer.as_default():
                tf.summary.scalar('loss',loss,step = s) #s for training session

        train_loop.set_description(f'Step {s}/{num_steps}, Loss {moving_loss:.5f}')

        if s != 0 and s % 50000 == 0:
            filename = f'weights-step{s:07d}-loss{moving_loss:.5f}.hdf5'
            model.save_weights(os.path.join(os.path.dirname(__file__), 'checkpoints_correction', filename))
            np.save(os.path.join(os.path.dirname(__file__), 'checkpoints_correction', f'{filename}_optimizer.npy'), optimizer.get_weights())

        if s != 0 and s % 10000 == 0:
            # perform validation
            errors = validation(model, valid_dataset)
            with val_summary_writer.as_default():
                for k, v in errors.items():
                    tf.summary.scalar(f'validation {k}-rmse', v, step=s)
            print(', '.join([f'{k}-step RMSE: {v}' for k, v in errors.items()]))


def main():

    train()


if __name__ == '__main__':
    main()
