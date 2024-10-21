import subprocess
import itertools
import csv
import re

# Fixed parameters
num_segments = 3
save_dir = "tune/"
random_seed = 42
action_dim = 5
max_steps = 12
batch_size = 512
latent_dist = "gaussian"
output_file = "tune/hyperparameter_sweep_results.csv"
demo_file = "trajectories/colours/15k_nopick"

# Hyperparameters to sweep
learning_rates = [ 1e-4, 1e-3 ]
hidden_dims = [ 128, 256, 512 ]
latent_dims = [ 8, 16, 32, 64, 128 ]
iterations = [ 100, 1000, 5000, 10000, 50000 ]
beta_bs = [  0.5, 1.0 ]
beta_zs = [  0.5, 1.0 ]
prior_rates = [ 3.0, 4.0, 5.0 ]




combinations = list(itertools.product(learning_rates, hidden_dims, latent_dims, iterations, beta_bs, beta_zs, prior_rates))
print(f"Total number of combinations: {len(combinations)}")


run_count = 0

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["run_id", "learning_rate", "hidden_dim", "latent_dim", "iterations", "beta_b", "beta_z", "prior_rate", "seg_acc", "skill_acc", "l2_distance"])


for lr, hidden_dim, latent_dim, iteration, bb, bz, pr  in itertools.product(learning_rates, hidden_dims, latent_dims, iterations, beta_bs, beta_zs, prior_rates):

    cmd = [
        "python3", f"train.py",
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
        "--action-dim", str(action_dim),
        "--max-steps", str(max_steps),
        "--beta-b", str(bb),
        "--beta-z", str(bz),
        "--prior-rate", str(pr),
        "--train-model"
    ]

    
    print(f"Running: {' '.join(cmd)}")

    try: 
        result = subprocess.run(cmd, capture_output=True, text=True)
        skill_acc, seg_acc, l2_dist = result.stdout.strip().split(" ")
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run_count, lr, hidden_dim, latent_dim, iteration, bb, bz, pr, seg_acc, skill_acc, l2_dist])
        file.close()  
    except Exception as e:
        print(f"Error with config: {' '.join(cmd)}")
        print(f"Exception: {e}")
    
    run_count += 1