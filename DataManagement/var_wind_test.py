from pathlib import Path
import subprocess as sp
import os
import json

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
                 self.triangles.append(list(map(lambda v: int(v.split('/')[0]) - 1, [v for v in tokens[1:]])))
                 continue

def get_handles(mesh, handle_positions):
    handles = []
    for handle_pos in handle_positions:
        for idx, vertex in enumerate(mesh.verticies):
            if vertex == handle_pos:
                handles.append(idx)
                break
    assert len(handles) == len(handle_positions)
    return handles
            

def run(arcsim_config, wind_config, arcsim_path, save_dir):
    mesh_path = arcsim_config["cloths"][0]["mesh"]
    with open(mesh_path, 'r') as f:
        mesh = OBJ(f)
    handles = arcsim_config["handles"][0]["nodes"]
    handle_positions = [mesh.verticies[n] for n in handles]
    for traj_id, keyframes in enumerate(wind_config['trajectories']):
        temp_config = arcsim_config
        # initial setup
        first, *rest = keyframes
        current_frame, current_wind = first

        temp_config['wind']['velocity'] = current_wind
        temp_config['end_frame'] = current_frame

        tmp_conf_path = os.path.join(save_dir, f'traj_{traj_id:04d}.json')
        with open(tmp_conf_path, 'w') as f:
            json.dump(temp_config, f)
        working_dir = os.path.join(save_dir, f'traj_{traj_id:04d}')
        sp.run([arcsim_path, "simulateoffline", tmp_conf_path, os.path.join(save_dir, f'traj_{traj_id:04d}')])
        prev_frame = current_frame
        while rest:
            first, *rest = rest
            current_frame, current_wind = first
            with open(os.path.join(working_dir, "conf.json"), 'r') as f:
                working_config = json.load(f)

            # Set next wind velocity and frame end
            working_config['wind']['velocity'] = current_wind
            working_config['end_frame'] = current_frame 
            
            # Find new handle indicies and set
            with open(os.path.join(working_dir, f'traj_{traj_id:04d}', f'{current_frame:04d}_00.obj')) as f:
                mesh = OBJ(f)
            handles = get_handles(mesh, handle_positions)
            working_config["handles"][0]["nodes"] = handles
            
            tmp_conf_path = os.path.join(save_dir, f'traj_{traj_id:04d}.json')
            with open(os.path.join(working_dir, "conf.json"), 'w') as f:
                json.dump(working_config, f)
            print(current_frame, current_wind)
            sp.run([arcsim_path, "resumeoffline", working_dir, str(prev_frame)])
            prev_frame = current_frame
        with open(os.path.join(working_dir, "wind.json"), 'w') as f:
            json.dump({"trajectories": keyframes}, f)


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'no_strainlimiting.json')
    with open(config_path, 'r') as f:
        arcsim_config = json.load(f)
    with open('wind.json', 'r') as f:
        wind_config = json.load(f)
    run(arcsim_config, wind_config, "../arcsim/bin/arcsim", "data")


if __name__ == "__main__":
    main()

