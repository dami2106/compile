#!/bin/bash


trajectory="trajectories/colours/15k"
model_dir="runs/50000e_15k_colours"

# Run the train.py script with the provided arguments
python train.py \
    --demo-file $trajectory \
    --save-dir $model_dir \