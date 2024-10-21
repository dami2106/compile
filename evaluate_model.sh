#!/bin/bash


trajectory="trajectories/colours/15k"
model_dir="runs/1000e_15k_colours"

# Run the train.py script with the provided arguments
python3 train.py \
    --demo-file $trajectory \
    --save-dir $model_dir \