#!/bin/bash

# Variables
iterations=5
learning_rate=0.001
hidden_dim=5 #256
latent_dim=2 #32
latent_dist="gaussian"  # Or "concrete"
batch_size=10
num_segments=3
demo_file="trajectories/colours/100_layered"
save_dir="runs/testing"
random_seed=42
train_model=true
state_dim=100
action_dim=4
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
    --state-dim $state_dim \
    --action-dim $action_dim \
    --max-steps $max_steps \
    $( [[ "$train_model" == true ]] && echo "--train-model" ) \