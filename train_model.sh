#!/bin/bash
# export QT_QPA_PLATFORM=offscreen
# export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# ldconfig -p | grep cuda

# Variables
iterations=5000
domain="colours"
learning_rate=0.0001
hidden_dim=128 #256
latent_dim=64 #32
latent_dist="gaussian"  # Or "concrete"
batch_size=256
num_segments=3
demo_file="trajectories/colours/100_nopick"
save_dir="runs/colours/static_colours"
random_seed=42
train_model=true
verbose=true
action_dim=4
max_steps=12
out_channels=8
kernel=3
stride=1


python3 train_colours.py \
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
    --action-dim $action_dim \
    --max-steps $max_steps \
    --out-channels $out_channels \
    --kernel $kernel \
    --stride $stride \
    $( [[ "$train_model" == true ]] && echo "--train-model" ) \
    $( [[ "$verbose" == true ]] && echo "--verbose" ) \


# 31	0.0001	128	32	5000	64	3	0.967626	0.411871	0.391249
