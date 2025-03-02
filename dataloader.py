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
    """Pads sequences using the last state instead of padding with zeros."""
    max_length = max(seq.shape[0] for seq in data_list)  # Find max sequence length

    padded_sequences = []
    for seq in data_list:
        pad_length = max_length - seq.shape[0]
        if pad_length > 0:
            last_state = seq[-1].unsqueeze(0)  # Get last state and expand dimensions
            padding = last_state.repeat(pad_length, *([1] * (seq.dim() - 1)))  # Repeat last state
            padded_seq = torch.cat([seq, padding], dim=0)  # Concatenate with original sequence
        else:
            padded_seq = seq  # No padding needed
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences)  # Stack into batch tensor