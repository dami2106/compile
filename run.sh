#!/bin/bash

# Variables
iterations=30000
learning_rate=0.001
hidden_dim=256
latent_dim=32
latent_dist="gaussian"  # Or "concrete"
batch_size=512
num_segments=3
demo_file="trajectories/colours/5k"
save_dir="runs/15000e_5k_colours"
random_seed=42
train_model=true
state_dim=11
action_dim=5
max_steps=12

# Run the train.py script with the provided arguments
python3 train.py \
    --iterations $iterations \
    --learning-rate $learning_rate \
    --hidden-dim $hidden_dim \
    --latent-dim $latent_dim \
    --latent-dist $latent_dist \
    --batch-size $batch_size \
    --num-segments $num_segments \
    --demo-file $demo_file \
    --save-dir $save_dir \
    --random-seed $random_seed \
    $( [[ "$train_model" == true ]] && echo "--train-model" ) \
    --state-dim $state_dim \
    --action-dim $action_dim \
    --max-steps $max_steps