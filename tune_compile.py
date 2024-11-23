import subprocess
import itertools
import csv
import re

# Fixed parameters
num_segments = 3
random_seed = 42
action_dim = 5
max_steps = 12
batch_size = 512
learning_rate=0.0001
hidden_dim=128
latent_dim=64
beta_b=0.5
beta_z=1.0
prior_rate=5.0

save_dir = "tune/"
latent_dist = "gaussian"
output_file = "tune/hyperparameter_sweep_results.csv"
demo_file = "trajectories/colours/"


# Hyperparameters to sweep

iterations = [ 50, 100, 1000, 5000, 10000, 50000 ]
files =      [ '100_nopick', '1000_nopick', '5000_nopick', '15000_nopick' ]



combinations = list(itertools.product(iterations, files))
print(f"Total number of combinations: {len(combinations)}")


run_count = 0

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    #L2 Dist	F1 Full	F1 Per	MIOU Full	MIOU Per	MOF Full	MOF Per
    writer.writerow(["run_id",  "iterations", "data", "l2_distance", "F1_full", "F1_per", "MIOU_full", "MIOU_per", "MOF_full", "MOF_per"])


for iteration, file_ in itertools.product(iterations, files):

    cmd = [
        "python3", f"train.py",
        "--iterations", str(iteration),
        "--learning-rate", str(learning_rate),
        "--hidden-dim", str(hidden_dim),
        "--latent-dim", str(latent_dim),
        "--latent-dist", latent_dist,
        "--batch-size", str(batch_size),
        "--num-segments", str(num_segments),
        "--demo-file", demo_file + file_,
        "--save-dir", save_dir + f"run_{run_count}",
        "--random-seed", str(random_seed),
        "--action-dim", str(action_dim),
        "--max-steps", str(max_steps),
        "--train-model",
        "--beta-b", str(beta_b),
        "--beta-z", str(beta_z),
        "--prior-rate", str(prior_rate),
    ]

    
    print(f"Running: {' '.join(cmd)}")

    try: 
        result = subprocess.run(cmd, capture_output=True, text=True)
        l2, f1_full, f1_per, miou_full, miou_per, mof_full, mof_per = result.stdout.strip().split(" ")


        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run_count, iteration, file_, l2, f1_full, f1_per, miou_full, miou_per, mof_full, mof_per])


        file.close()  
    except Exception as e:
        print(f"Error with config: {' '.join(cmd)}")
        print(f"Exception: {e}")
    
    print()
    print()
    
    run_count += 1