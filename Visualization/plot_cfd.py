# -*- coding: utf-8 -*-
import os
import pickle
import argparse
from pathlib import Path


import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import tri as mtri



def plot_cfd_color(data, filename):
    figure =plt.figure(figsize = (12,7))
    axis = figure.add_subplot(211)
    axis2 = figure.add_subplot(212)
    skip = 1
    num_steps = data[0]['true_velocity'].shape[0]
    num_frames = len(data) * num_steps // skip

    # compute bounds
    bounds = []
    for trajectory in data:
        bb_min = trajectory['true_velocity'].min(axis=(0, 1))
        bb_max = trajectory['true_velocity'].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))


    def animate(frame):
         step = (frame*skip) % num_steps
         traj = (frame*skip) // num_steps
         axis.cla()
         axis.set_aspect('equal')
         axis.set_axis_off()
         vmin, vmax = bounds[traj]
         pos = data[traj]['mesh_pos'][step]
         faces = data[traj]['cells'][step]
         velocity = data[traj]['pred_velocity'][step]
         triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
         axis.tripcolor(triang, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
         axis.triplot(triang, 'ko-', ms=0.5, lw=0.3)
         axis.set_title("Predicted")

         step = (frame*skip) % num_steps
         traj = (frame*skip) // num_steps
         axis2.cla()
         axis2.set_aspect('equal')
         axis2.set_axis_off()
         vmin, vmax = bounds[traj]
         pos = data[traj]['mesh_pos'][step]
         faces = data[traj]['cells'][step]
         velocity = data[traj]['true_velocity'][step]
         triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
         axis2.tripcolor(triang, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
         axis2.triplot(triang, 'ko-', ms=0.5, lw=0.3)
         axis2.set_title("Ground Truth")

         figure.suptitle(f"Time step: {frame}")

         return figure,

    ani = animation.FuncAnimation(figure, animate, frames=num_frames, interval=30)

    ani.save(filename)


def plot_cfd_quiver(data, filename):
    figure =plt.figure(figsize = (12,7))
    axis = figure.add_subplot(211)
    axis2 = figure.add_subplot(212)
    skip = 1
    num_steps = data[0]['true_velocity'].shape[0]
    num_frames = len(data) * num_steps // skip

    # compute bounds
    bounds = []
    for trajectory in data:
        bb_min = trajectory['true_velocity'].min(axis=(0, 1))
        bb_max = trajectory['true_velocity'].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))


    def animate(frame):
         step = (frame*skip) % num_steps
         traj = (frame*skip) // num_steps
         axis.cla()
         axis.set_aspect('equal')
         axis.set_axis_off()
         vmin, vmax = bounds[traj]
         pos = data[traj]['mesh_pos'][step]
         faces = data[traj]['cells'][step]
         velocity = data[traj]['pred_velocity'][step]
         triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
         axis.quiver(pos[:, 0], pos[:, 1], velocity[:, 0], velocity[:, 1])
         axis.triplot(triang, 'ko-', ms=0.5, lw=0.3)
         axis.set_title("Predicted")

         step = (frame*skip) % num_steps
         traj = (frame*skip) // num_steps
         axis2.cla()
         axis2.set_aspect('equal')
         axis2.set_axis_off()
         vmin, vmax = bounds[traj]
         pos = data[traj]['mesh_pos'][step]
         faces = data[traj]['cells'][step]
         velocity = data[traj]['true_velocity'][step]
         triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

         axis2.quiver(pos[:, 0], pos[:, 1], velocity[:, 0], velocity[:, 1])
         axis2.triplot(triang, 'ko-', ms=0.5, lw=0.3)
         axis2.set_title("Ground Truth")

         figure.suptitle(f"Time step: {frame}")

         return figure,

    ani = animation.FuncAnimation(figure, animate, frames=num_frames, interval=30)

    # plt.show()
    ani.save(filename)


def generate_all():
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'results', 'cfd', 'og_long-step4200000-loss0.01691.hdf5')
    output_path = os.path.join(os.path.dirname(__file__), 'animations', 'cfd', 'og_long-step4200000-loss0.01691.hdf5')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    for i in range(10, 100):
        with open(os.path.join(results_path, f'{i:03d}.eval'), 'rb') as f:
            data = pickle.load(f)
        plot_cfd_color([data], os.path.join(output_path, f'c{i:03d}.mp4'))
        plot_cfd_quiver([data], os.path.join(output_path, f'q{i:03d}.mp4'))


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
