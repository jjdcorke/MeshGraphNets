import os
import json
import sys
import shutil



def perform_clean():
    save_dir = os.path.join(os.path.dirname(__file__), 'test2')
    items = os.listdir(save_dir)

    for item in items:
        if item.endswith('.json'):
            os.remove(os.path.join(save_dir, item))

    items = [item for item in items if not item.endswith('.json')]

    successes = 0
    failures = 0
    for item in items:
        trajectory_dir = os.path.join(save_dir, item)
        trajectory_items = os.listdir(trajectory_dir)
        
        with open(os.path.join(trajectory_dir, 'conf.json'), 'r') as f:
            config = json.load(f)

        required_num = config['end_frame'] + 3
        if len(trajectory_items) != required_num:
            print(item, 'failed')
            failures += 1

            # delete the failed directory
            shutil.rmtree(trajectory_dir)
        else:
            successes += 1

    print('Total successes:', successes)
    print('Total failures:', failures)
    

def main():
    perform_clean()


if __name__ == "__main__":
    main()

