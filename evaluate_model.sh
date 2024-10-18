#!/bin/bash


trajectory="trajectories/colours/15k_layered"
model_dir="runs/colours/cnn_15k_50ke"

# Run the train.py script with the provided arguments
python3 train.py \
    --demo-file $trajectory \
    --save-dir $model_dir \