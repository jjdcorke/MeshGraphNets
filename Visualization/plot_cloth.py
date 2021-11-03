import pickle
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import argparse



def plot_cloth(data, filename):
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    num_frames = data["world_pos"].shape[0]
    upper_bound = np.amax(data["world_pos"], axis=(0,1))
    lower_bound= np.amin(data["world_pos"], axis=(0,1))
    def animate(frame):
        axis.cla()
        axis.set_xlim([lower_bound[0], upper_bound[0]])
        axis.set_ylim([lower_bound[1], upper_bound[1]])
        axis.set_zlim([lower_bound[2], upper_bound[2]])
        axis.autoscale(False)
        positions = data['world_pos'][frame]
        faces = data["cells"][frame]
        axis.plot_trisurf(positions[:,0], positions[:, 1], faces, positions[:, 2])
        return figure,

    ani = animation.FuncAnimation(figure, animate, frames=num_frames)
    ani.save(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("-o", "--output", help="output file")
    args = parser.parse_args()
    if (args.output == None):
        args.output = "out.mp4"
    data = pickle.load(open(args.datafile, "rb"))
    plot_cloth(data, args.output)



if __name__ == "__main__":
    main()
