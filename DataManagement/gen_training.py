from pathlib import Path
import subprocess as sp
import os
import json
import time
import sys


import numpy as np



def run(traj_id, cpu_ids, config, arcsim_path, save_dir):

    # generate random wind velocity
    wind_speed = np.random.uniform(0, 20)
    wind_direction = np.random.rand(3)
    wind_direction = wind_direction / np.linalg.norm(wind_direction)
    wind_velocity = (wind_speed * wind_direction).tolist()

    # generate random rotation
    rotation_angle = np.random.randint(0, 360)
    rotation_axis = np.random.rand(3)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation = [rotation_angle] + rotation_axis.tolist()

    config['wind']['velocity'] = wind_velocity
    config['cloths'][0]['transform']['rotate'] = rotation

    tmp_conf_path = os.path.join(save_dir, f'traj_{traj_id:04d}.json')
    with open(tmp_conf_path, 'w') as f:
        json.dump(config, f)

    process = sp.Popen(['taskset', '-c', ','.join([str(cpu) for cpu in cpu_ids]), arcsim_path,
                        'simulateoffline', tmp_conf_path, os.path.join(save_dir, f'traj_{traj_id:04d}')],
                       stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)

    return process



def run_all_trajectories():
    arcsim_path = '/home/eydjiang/arcsim-0.2.1/bin/arcsim'
    save_dir = os.path.join(os.path.dirname(__file__), 'test2')
    num_cores = 30
    num_trajectories = 180

    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'no_strainlimiting.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    cpu_to_process_map = {i: None for i in range(num_cores)}

    #for i in range(num_trajectories):
    for i in range(1000, 1000 + num_trajectories):

        # try to find an available cpu
        for cpu_id, process in cpu_to_process_map.items():
            if process is None:
                new_process = run(i, [cpu_id], config, arcsim_path, save_dir)
                break
        else:
            # no cpus are available, so wait until one is
            while True:
                for cpu_id, process in cpu_to_process_map.items():
                    if process.poll() is not None:
                        # process for this cpu has terminated
                        print('Process return code:', process.returncode)
                        print('Output:', process.stdout.read())
                        break
                else:
                    time.sleep(0.1)
                    continue
                break

            new_process = run(i, [cpu_id], config, arcsim_path, save_dir)

        print(f'Running trajectory {i} on cpu {cpu_id}...')
        cpu_to_process_map[cpu_id] = new_process

    # wait for everything to finish
    for cpu_id, process in cpu_to_process_map.items():
        process.wait()
        print('Process return code:', process.returncode)
        print('Output:', process.stdout.read())


def main():
    run_all_trajectories()


if __name__ == "__main__":
    main()

