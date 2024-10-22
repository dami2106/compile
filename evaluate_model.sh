#!/bin/bash

model_dir="runs/treasure/10k_20ke"

python3 train_treasure.py \
    --save-dir $model_dir 
