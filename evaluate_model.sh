#!/bin/bash

model_dir="runs/treasure/5000_5_5ke"

python3 train_treasure.py \
    --save-dir $model_dir 
