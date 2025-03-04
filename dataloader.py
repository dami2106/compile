import os
import numpy as np
import torch

def load_trajectories(features_path, actions_path, groundTruth_path):
    """Loads all trajectories and returns lists of tensors for states, actions, and ground truth.
    
    Each episode is padded to the length of the longest episode using the last available
    state, action, and ground truth.
    """
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
    
    # Determine the maximum episode length among all loaded episodes
    max_length = max([states.shape[0] for states in state_tensors]) if state_tensors else 0

    # Pad each episode to the maximum length
    for i in range(len(state_tensors)):
        current_length = state_tensors[i].shape[0]
        if current_length < max_length:
            pad_count = max_length - current_length

            # Pad states: repeat the last state pad_count times
            last_state = state_tensors[i][-1].unsqueeze(0)
            pad_states = last_state.repeat(pad_count, 1)
            state_tensors[i] = torch.cat([state_tensors[i], pad_states], dim=0)

            # Pad actions: repeat the last action pad_count times
            # Handles both 1D and multi-dimensional actions
            last_action = action_tensors[i][-1].unsqueeze(0)
            # Create repeat pattern based on the tensor's dimensions
            repeat_pattern = [pad_count] + [1] * (action_tensors[i].dim() - 1)
            pad_actions = last_action.repeat(*repeat_pattern)
            action_tensors[i] = torch.cat([action_tensors[i], pad_actions], dim=0)

            # Pad ground_truth: append the last truth pad_count times
            last_truth = ground_truth[i][-1]
            pad_truths = [last_truth] * pad_count
            ground_truth[i] = ground_truth[i] + pad_truths

    return state_tensors, action_tensors, ground_truth