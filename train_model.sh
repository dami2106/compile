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
batch_size=512
num_segments=3
demo_file="trajectories/colours/15k_layered_static_rand"
save_dir="runs/colours/5ke_15k_colours_static_rand_2"
# save_dir="runs/colours/test2"
random_seed=45
train_model=true
action_dim=5
max_steps=12
out_channels=8
kernel=3
stride=1

# 0.0001	128	64	5000	8	3	0.971429	0.424405	0.406127

# Run the train.py script with the provided arguments
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


# 31	0.0001	128	32	5000	64	3	0.967626	0.411871	0.391249