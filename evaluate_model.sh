#!/bin/bash

# export QT_QPA_PLATFORM=offscreen
# export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# ldconfig -p | grep cuda

model_dir="runs/colours/testing_new"

python3 train_colours.py \
    --save-dir $model_dir 
