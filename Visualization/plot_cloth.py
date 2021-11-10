import os
import pickle
import argparse
from pathlib import Path


import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt




def plot_cloth(data, filename):
    figure = plt.figure()
    axis = figure.add_subplot(121, projection="3d")
    axis2 = figure.add_subplot(122, projection="3d")
    figure.set_figheight(6)
    figure.set_figwidth(12)

    num_frames = data["pred_world_pos"].shape[0]
    upper_bound = np.amax(data["pred_world_pos"], axis=(0,1))
    lower_bound= np.amin(data["pred_world_pos"], axis=(0,1))
    def animate(frame):
        axis.cla()
        axis.set_xlim([lower_bound[0], upper_bound[0]])
        axis.set_ylim([lower_bound[1], upper_bound[1]])
        axis.set_zlim([lower_bound[2], upper_bound[2]])
        axis.autoscale(False)
        positions = data['pred_world_pos'][frame]
        faces = data["cells"][frame]
        axis.plot_trisurf(positions[:,0], positions[:, 1], faces, positions[:, 2])
        axis.set_title('Predicted')

        axis2.cla()
        axis2.set_xlim([lower_bound[0], upper_bound[0]])
        axis2.set_ylim([lower_bound[1], upper_bound[1]])
        axis2.set_zlim([lower_bound[2], upper_bound[2]])
        axis2.autoscale(False)
        positions = data['true_world_pos'][frame]
        faces = data["cells"][frame]
        axis2.plot_trisurf(positions[:,0], positions[:, 1], faces, positions[:, 2])
        axis2.set_title('Ground Truth')

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
        plot_cloth(data, os.path.join(output_path, f'{i:03d}.mp4'))


def avg_rmse():
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'results')
    results_prefixes = ['weights-step2700000-loss0.0581.hdf5', 'weights-step2300000-loss0.04908.hdf5']

    for prefix in results_prefixes:
        all_errors = []
        for i in range(100):
            with open(os.path.join(results_path, f'{prefix}_{i:03d}.eval'), 'rb') as f:
                data = pickle.load(f)
                all_errors.append(data['errors'])

        keys = list(all_errors[0].keys())
        all_errors = {k: np.array([errors[k] for errors in all_errors]) for k in keys}

        for k, v in all_errors.items():
            print(prefix, k, np.mean(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("-o", "--output", help="output file")
    args = parser.parse_args()
    if (args.output == None):
        args.output = "out.mp4"

    with open(args.datafile, "rb") as f:
        data = pickle.load(f)

    plot_cloth(data, args.output)



if __name__ == "__main__":
    generate_all()
    # main()
