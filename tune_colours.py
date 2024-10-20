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
action_dim = 4
max_steps = 12
batch_size = 512
latent_dist = "gaussian"
output_file = "tune/hyperparameter_sweep_results.csv"

# Hyperparameters to sweep
# learning_rates = [ 1e-4, 1e-3 ]
# hidden_dims = [ 128, 256, 512 ]
# latent_dims = [ 16, 32, 64, 128 ]
# iterations = [ 100, 1000, 5000, 10000, 50000 ]
# out_channels = [ 8, 16, 32, 64 ]
# kernels = [ 3 ]

learning_rates = [ 1e-4 ]
hidden_dims = [ 128 ]
latent_dims = [ 16 ]
iterations = [ 2, 3 ]
out_channels = [ 8 ]
kernels = [ 3 ]



run_count = 0

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["run_id", "learning_rate", "hidden_dim", "latent_dim", "iterations", "out_channels", "kernel", "seg_acc", "skill_acc", "l2_distance"])


for lr, hidden_dim, latent_dim, iteration, out_channel, kernel  in itertools.product(learning_rates, hidden_dims, latent_dims, iterations, out_channels, kernels):

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
        "--action-dim", str(action_dim),
        "--max-steps", str(max_steps),
        "--out-channels", str(out_channel),
        "--kernel", str(kernel),
        "--train-model"
    ]

    
    print(f"Running: {' '.join(cmd)}")

    try: 
        result = subprocess.run(cmd, capture_output=True, text=True)
        skill_acc, seg_acc, l2_dist = result.stdout.strip().split(" ")
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run_count, lr, hidden_dim, latent_dim, iteration, out_channel, kernel, seg_acc, skill_acc, l2_dist])
        file.close()  
    except Exception as e:
        print(f"Error with config: {' '.join(cmd)}")
        print(f"Exception: {e}")
    
    run_count += 1
