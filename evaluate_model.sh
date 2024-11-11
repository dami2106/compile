#!/bin/bash

export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
ldconfig -p | grep cuda

model_dir="runs/colours/50ke_15k_colours_static_rand_ra"

python3 train_colours.py \
    --save-dir $model_dir 
