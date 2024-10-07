#!/bin/bash


trajectory="trajectories/colours/15k_square"
model_dir="runs/test_cnn"

# Run the train.py script with the provided arguments
python3 train.py \
    --demo-file $trajectory \
    --save-dir $model_dir \