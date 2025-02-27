import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def load_trajectories(features_path, actions_path, groundTruth_path):
    """Loads all trajectories and returns lists of tensors for states, actions, and ground truth."""
    state_tensors, action_tensors, ground_truth = [], [], []
    episode_files = sorted(os.listdir(features_path))

    for file in episode_files:
        if file.endswith(".npy"):
            state_file = os.path.join(features_path, file)
            action_file = os.path.join(actions_path, file)
            ground_truth_file = os.path.join(groundTruth_path, file[:-4])
            
            if os.path.exists(action_file) and os.path.exists(ground_truth_file):
                states = torch.tensor(np.load(state_file), dtype=torch.float32)
                actions = torch.tensor(np.load(action_file), dtype=torch.long)
                
                # Ground truth file is a text file with each item on a new line
                with open(ground_truth_file, 'r') as f:
                    truths = f.readlines()
                    truths = [truth.strip() for truth in truths]

                state_tensors.append(states)
                action_tensors.append(actions)
                ground_truth.append(truths)
    
    return state_tensors, action_tensors, ground_truth

def pad_and_batch(data_list):
    """Pads sequences for batch processing."""
    return pad_sequence(data_list, batch_first=True, padding_value=0)