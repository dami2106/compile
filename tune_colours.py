import subprocess
import itertools
import csv
import re

# Fixed parameters
domain = "colours"
num_segments = 3
demo_file = "trajectories/colours/15k_layered"
save_dir = "tune/colours/"
random_seed = 42
state_dim = 3
action_dim = 4
max_steps = 12
batch_size = 512
latent_dist = "gaussian"
output_file = "tune/hyperparameter_sweep_results.csv"

# Hyperparameters to sweep
learning_rates = [1e-4, 1e-3]
hidden_dims = [64, 128, 256, 512, 1024]
latent_dims = [8, 16, 32, 64, 128]
# iterations = [5000, 10000, 20000, 50000]
iterations = [2]

run_count = 0

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["run_id", "learning_rate", "hidden_dim", "latent_dim", "iterations",  "seg_acc", "skill_acc"])


    for lr, hidden_dim, latent_dim, iteration in itertools.product(learning_rates, hidden_dims, latent_dims, iterations):

        cmd = [
            "python3", f"train_{domain}.py",
            "--iterations", str(iteration),
            "--learning-rate", str(lr),
            "--hidden-dim", str(hidden_dim),
            "--latent-dim", str(latent_dim),
            "--latent-dist", latent_dist,
            "--batch-size", str(batch_size),
            "--num-segments", str(num_segments),
            "--demo-file", demo_file,
            "--save-dir", save_dir + f"run_{run_count}",
            "--random-seed", str(random_seed),
            "--state-dim", str(state_dim),
            "--action-dim", str(action_dim),
            "--max-steps", str(max_steps),
            "--train-model"
        ]

        run_count += 1
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        print("done")
        output = result.stdout