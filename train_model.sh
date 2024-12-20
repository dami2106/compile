#!/bin/bash

# Variables
iterations=10000
learning_rate=0.0001
hidden_dim=128
latent_dim=64
latent_dist="gaussian"  # Or "concrete"
batch_size=512
num_segments=3
demo_file="trajectories/colours/100_nopick"
save_dir="runs/testing4"
random_seed=42
train_model=true
verbose=true
state_dim=11
action_dim=4
max_steps=12
beta_b=0.5
beta_z=1.0
prior_rate=5.0

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
    --beta-b $beta_b \
    --beta-z $beta_z \
    --prior-rate $prior_rate \
    $( [[ "$train_model" == true ]] && echo "--train-model" ) \
    $( [[ "$verbose" == true ]] && echo "--verbose" ) \