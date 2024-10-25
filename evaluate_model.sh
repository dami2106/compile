#!/bin/bash

model_dir="runs/50000e_15k_colours"

# Run the train.py script with the provided arguments
python3 train.py \
    --save-dir $model_dir \