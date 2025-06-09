from config import (
    BACKGROUNDS_DIR, TARGET_ASSETS_DIR, OUTPUT_DIR
)

import argparse

def main(task, num_samples):
    if task == 'gate':
        pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Synthetic data generator for RoboSub tasks")
    parser.add_argument('--task', type=str, required=True, help="Which task to generate data for: gate or torpedo")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()

    main(args.task, args.num_samples)