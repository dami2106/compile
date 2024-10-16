#!/bin/bash

# Variables
iterations=20000
learning_rate=0.001
hidden_dim=256 #256
latent_dim=32 #32
latent_dist="gaussian"  # Or "concrete"
batch_size=128
num_segments=5
demo_file="trajectories/treasure/10000_30"
save_dir="runs/treasure/10k_20ke"
random_seed=42
train_model=true
state_dim=3
action_dim=5
max_steps=30

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
    --state-dim $state_dim \
    --action-dim $action_dim \
    --max-steps $max_steps \
    $( [[ "$train_model" == true ]] && echo "--train-model" ) \