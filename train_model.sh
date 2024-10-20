#!/bin/bash

# Variables
iterations=100
domain="colours"
learning_rate=0.0001
hidden_dim=128 #256
latent_dim=16 #32
latent_dist="gaussian"  # Or "concrete"
batch_size=512
num_segments=3
demo_file="trajectories/colours/15k_layered"
save_dir="runs/colours/test"
random_seed=42
train_model=true
action_dim=4
max_steps=12
out_channels=8
kernel=3
stride=1

# Run the train.py script with the provided arguments
python3 train_$domain.py \
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


# Error with config: python3 train_colours.py --iterations 100 
# --learning-rate 0.0001 --hidden-dim 128 --latent-dim 16
#  --latent-dist gaussian --batch-size 512 --num-segments 3 
#  --demo-file trajectories/colours/15k_layered --save-dir tune/colours/run_0 
#  --random-seed 42 --action-dim 4 --max-steps 12 --out-channels 8 --kernel 2 --train-model