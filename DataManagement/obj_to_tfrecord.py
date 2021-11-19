import json
from pathlib import Path
import tensorflow as tf
import argparse

class OBJ:
    def __init__(self, fp):
        self.verticies = []
        self.texcoords = []
        self.triangles = []
        for line in fp:
            tokens = line.split()
            if not tokens:
                continue
            elif tokens[0] == '#':
                continue
            elif tokens[0] == 'v':
                self.verticies.append([float(v) for v in tokens[1:]])
                continue
            elif tokens[0] == "vt":
                self.texcoords.append([float(v) for v in tokens[1:]])
                continue
            elif tokens[0] == 'f':
                 self.triangles.append(list(map(lambda v: int(v.split('/')[0]), [v for v in tokens[1:]])))
                 continue


def process_trajectory(path):
    json_file = Path(path).glob("*.json")
    for file in json_file:
        with open(file, 'r') as fp:
            conf = json.load(fp)
    obj_files = sorted(Path(path).glob("*.obj"))
    wind_velocity = []
    world_pos = []
    node_type = []
    for obj in obj_files:
        with open(obj, 'r') as fp:
            mesh = OBJ(fp)
            world_pos.append(mesh.verticies)
            wind_velocity.append(conf["wind"]["velocity"])
            nodes = [[0]] * len(mesh.verticies) # NodeType Normal
            for handle in conf["handles"][0]["nodes"]:
                nodes[handle] = [3] # NodeType HANDLE
            node_type.append(nodes) 
    world_pos = tf.constant(world_pos)
    wind_velocity = tf.constant(wind_velocity)
    node_type = tf.constant(node_type)
    return world_pos, wind_velocity, node_type


def create_record(data_path, template_path, out_file="train.tfrecord", meta_file="meta.json"):
    with open(template_path, 'r') as fp:
        template = json.load(fp)
    mesh_path = template["cloths"][0]["mesh"]
    with open(mesh_path, 'r') as fp:
        mesh = OBJ(fp)
    cells = tf.constant([mesh.triangles])
    mesh_pos = tf.constant([mesh.verticies])
    with tf.io.TFRecordWriter(out_file) as writer:
        for i, traj_dir in enumerate(Path(data_path).iterdir()):
            world_pos, wind_velocity, node_type = process_trajectory(traj_dir)
            feature = { 
                        "cells" : tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(cells, [-1]))),
                        "mesh_pos" : tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(mesh_pos, [-1]))),
                        "node_type" : tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(node_type, [-1]))),
                        "world_pos" : tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(world_pos, [-1]))),
                        "wind_velocity" : tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(wind_velocity, [-1])))
                        }
            proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(proto.SerializeToString())
            print(i)

    features = {
    "cells": {
      "type": "static",
      "shape": cells.shape.as_list(),
      "dtype": "int32"
    },
    "mesh_pos": {
      "type": "static",
      "shape": mesh_pos.shape.as_list(),
      "dtype": "float32"
    },
    "node_type": {
      "type": "dynamic",
      "shape": node_type.shape.as_list(),
      "dtype": "int32"
    },
    "world_pos": {
      "type": "dynamic",
      "shape": world_pos.shape.as_list(),
      "dtype": "float32"
      },
    "wind_velocity": {
        "type": "dynamic",
        "shape": wind_velocity.shape.as_list(),
        "dtype": "float32"
        }
    }

    meta = {"simulator" : "arcsim",
            "dt" : template["frame_time"],
            "field_names" : ["cells", "mesh_pos", "node_type", "world_pos", "wind_speed"],
            "features" : features
            }
    with open(meta_file, 'w') as fp:
        json.dump(meta, fp, indent='\t')

def main():
    parser = argparse.ArgumentParser(description="Convert obj files produced by arcsim to a tfrecord for training.")
    parser.add_argument("data_dir", help="path to the parent directory of the trajectory directories")
    parser.add_argument("template_conf", help="path to the template arcsim config file")
    parser.add_argument("out_file", nargs='?', default="train.tfrecord", help="name of the produced tensorflow record file (default: train.tfrecord)")
    parser.add_argument("out_meta", nargs='?', default="meta.json", help="name of the produced metadata json file (default: meta.json)")
    args = parser.parse_args()
    create_record(args.data_dir, args.template_conf, out_file=args.out_file, meta_file=args.out_meta)



if __name__ == "__main__":
    main()
