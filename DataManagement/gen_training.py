from pathlib import Path
import subprocess as sp
import json
import random



def main():
    path = Path()
    data = path / "data" 
    config = path / "flag_template.json"
    with config.open() as fp:
        conf = json.load(fp)

    for i in range(1000):
        rand_velocity = [random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)]
        conf["wind"]["velocity"] = rand_velocity
        tmp_conf = path / "flag.json"
        with tmp_conf.open('w') as fp:
            json.dump(conf, fp)
        print("Generating trajectory #{}".format(i))
        sp.run(["../arcsim/bin/arcsim", "simulateoffline", "flag.json", "data/traj_{0:04d}".format(i) ])
            

        



if __name__ == "__main__":
    main()
