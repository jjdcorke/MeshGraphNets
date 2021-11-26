import os
import json
from functools import partial

import tensorflow as tf

from common import NodeType


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


def add_targets(trajectory, fields):
    out = {}
    for key, val in trajectory.items():
        out[key] = val[1:-1]
        if key in fields:
            out[f'prev|{key}'] = val[0:-2]
            out[f'target|{key}'] = val[2:]
    return out


def add_corrections(trajectory, fields):
    out = {}
    for key, val in trajectory.items():
        out[key] = val[1:-3]
        if key in fields:
            out[f'prev|{key}'] = val[0:-4]
            out[f'target1|{key}'] = val[2:-2]
            out[f'target2|{key}'] = val[3:-1]
            out[f'target3|{key}'] = val[4:]
    return out


def load_dataset_train(path, split, fields):
    with open(os.path.join(path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    dataset = tf.data.TFRecordDataset(os.path.join(path, f'{split}.tfrecord'))
    dataset = dataset.map(partial(parse, meta=meta), num_parallel_calls=8)
    dataset = dataset.prefetch(1)

    dataset = dataset.map(partial(add_corrections, fields=fields), num_parallel_calls=8)

    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    dataset = dataset.repeat(None)
    dataset = dataset.shuffle(10000)

    return dataset


def load_dataset_eval(path, split, fields):
    with open(os.path.join(path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    dataset = tf.data.TFRecordDataset(os.path.join(path, f'{split}.tfrecord'))
    dataset = dataset.map(partial(parse, meta=meta), num_parallel_calls=8)
    dataset = dataset.prefetch(1)

    dataset = dataset.map(partial(add_targets, fields=fields), num_parallel_calls=8)

    return dataset


def main():
    load_dataset_train(os.path.join('data', 'flag_simple'), 'valid', fields=['world_pos'])


if __name__ == '__main__':
    main()
