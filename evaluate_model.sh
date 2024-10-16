#!/bin/bash


trajectory="trajectories/colours/15k_omnp"
model_dir="runs/50000e_15k_colours_omnp_bigmodel"

# Run the train.py script with the provided arguments
python3 train.py \
    --demo-file $trajectory \
    --save-dir $model_dir \