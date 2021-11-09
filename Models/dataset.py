import os
import json
from functools import partial

import tensorflow as tf

from common import NodeType

@tf.function
def parse(proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)

    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')

        out[key] = data

    return out

@tf.function
def add_targets(trajectory, fields, add_history):
    out = {}
    for key, val in trajectory.items():
        out[key] = val[1:-1]
        if key in fields:
            out[f'target|{key}'] = val[2:]
            if add_history:
                out[f'prev|{key}'] = val[0:-2]
    return out

@tf.function
def add_noise(frame, fields, scale, gamma):
    if scale == 0:
        return frame

    for field in fields:
        noise = tf.random.normal(frame[field].shape, stddev=scale, dtype=tf.float32)

        # don't apply noise to boundary nodes
        mask = tf.equal(frame['node_type'], NodeType.NORMAL)
        noise = tf.where(mask, noise, tf.zeros_like(noise))

        frame[field] += noise
        frame[f'target|{field}'] += (1.0 - gamma) * noise

    return frame

@tf.function
def load_dataset_train(path, split, fields, add_history, noise_scale, noise_gamma):
    with open(os.path.join(path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    dataset = tf.data.TFRecordDataset(os.path.join(path, f'{split}.tfrecord'))
    dataset = dataset.map(partial(parse, meta=meta), num_parallel_calls=8)
    dataset = dataset.prefetch(1)

    dataset = dataset.map(partial(add_targets, fields=fields, add_history=add_history), num_parallel_calls=8)

    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    dataset = dataset.repeat(None)
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(partial(add_noise, fields=fields, scale=noise_scale, gamma=noise_gamma), num_parallel_calls=8)

    return dataset

@tf.function
def load_dataset_eval(path, split, fields, add_history):
    with open(os.path.join(path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    dataset = tf.data.TFRecordDataset(os.path.join(path, f'{split}.tfrecord'))
    dataset = dataset.map(partial(parse, meta=meta), num_parallel_calls=8)
    dataset = dataset.prefetch(1)

    dataset = dataset.map(partial(add_targets, fields=fields, add_history=add_history), num_parallel_calls=8)

    return dataset

@tf.function
def main():
    load_dataset_train(os.path.join('data', 'flag_simple'), 'valid', fields=['world_pos'],
                 add_history=True, noise_scale=0.003, noise_gamma=0.1)


if __name__ == '__main__':
    main()
