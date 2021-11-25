# -*- coding: utf-8 -*-
import os
import pickle
import argparse
from pathlib import Path


import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import tri as mtri



def plot_cfd(data, filename):
    figure =plt.figure(figsize = (12,6))
    axis = figure.add_subplot(121)
    axis2 = figure.add_subplot(122)
    skip = 10
    num_steps = data[0]['gt_velocity'].shape[0]
    num_frames = len(data) * num_steps // skip

  # compute bounds
    bounds = []
    for trajectory in data:
        bb_min = trajectory['gt_velocity'].min(axis=(0, 1))
        bb_max = trajectory['gt_velocity'].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))
   
    
    def animate(frame):
         step = (frame*skip) % num_steps
         traj = (frame*skip) // num_steps
         axis.cla()
         axis.set_aspect('equal')
         axis.set_axis_off()
         vmin, vmax = bounds[traj]
         pos = data[traj]['mesh_pos'][step]
         faces = data[traj]['faces'][step]
         velocity = data[traj]['pred_velocity'][step]
         triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
         axis.tripcolor(triang, velocity[:, 0], velocity[:, 1], vmin=vmin[0], vmax=vmax[0])
         axis.triplot(triang, 'ko-', ms=0.5, lw=0.3)
         axis.set_title("Predicted")
         
         step = (frame*skip) % num_steps
         traj = (frame*skip) // num_steps
         axis2.cla()
         axis2.set_aspect('equal')
         axis2.set_axis_off()
         vmin, vmax = bounds[traj]
         pos = data[traj]['mesh_pos'][step]
         faces = data[traj]['faces'][step]
         velocity = data[traj]['gt_velocity'][step]
         triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
         axis2.tripcolor(triang, velocity[:, 0], velocity[:, 1], vmin=vmin[0], vmax=vmax[0])
         axis2.triplot(triang, 'ko-', ms=0.5, lw=0.3)
         axis2.set_title("Ground Truth")
         
         figure.suptitle(f"Time step: {frame}")
         
         return figure,

    ani = animation.FuncAnimation(figure, animate, frames=num_frames, interval=100)

    ani.save(filename)


def generate_all():
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'results', 'weights-step2300000-loss0.04908.hdf5')
    output_path = os.path.join(os.path.dirname(__file__), 'animations', 'weights-step2300000-loss0.04908.hdf5')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    for i in range(100):
        with open(os.path.join(results_path, f'{i:03d}.eval'), 'rb') as f:
            data = pickle.load(f)
        plot_cfd(data, os.path.join(output_path, f'{i:03d}.mp4'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("-o", "--output", help="output file")
    args = parser.parse_args()
    if (args.output == None):
        args.output = "out.mp4"

    with open(args.datafile, "rb") as f:
        data = pickle.load(f)

    plot_cfd(data, args.output)



if __name__ == "__main__":
    generate_all()
    # main()

