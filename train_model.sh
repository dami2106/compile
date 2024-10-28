#!/bin/bash
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
ldconfig -p | grep cuda

# Variables
iterations=5000
domain="colours"
learning_rate=0.0001
hidden_dim=128 #256
latent_dim=64 #32
latent_dist="gaussian"  # Or "concrete"
batch_size=256
num_segments=3
demo_file="trajectories/colours/15k_layered"
save_dir="runs/colours/15k_colours_5ke_plots_new_2"
# save_dir="runs/colours/test2"
random_seed=42
train_model=true
action_dim=5
max_steps=12
out_channels=8
kernel=3
stride=1

# Run the train.py script with the provided arguments
python3 train_colours_vis.py \
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


