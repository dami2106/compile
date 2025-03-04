import numpy as np
import glob
import os 
import json 
import torch 

def load_data(args, device):
    state_files = glob.glob(f"{args.demo}/features/*.npy")
    state_dict = {}
    for file in state_files:
        key = os.path.splitext(os.path.basename(file))[0]  # e.g., "episode_0"
        state_dict[key] = np.load(file)

    # Build dictionary for actions
    action_files = glob.glob(f"{args.demo}/actions/*.npy")
    action_dict = {}
    for file in action_files:
        key = os.path.splitext(os.path.basename(file))[0]
        action_dict[key] = np.load(file)

    # Build dictionary for ground truths
    ground_truth_files = glob.glob(f"{args.demo}/groundTruth/*")
    ground_truth_dict = {}
    for file in ground_truth_files:
        # If the ground truth files don't have an extension, key will be the full name.
        key = os.path.splitext(os.path.basename(file))[0]
        with open(file) as f:
            ground_truth_dict[key] = f.read().splitlines()

    # Use the keys from state_dict (or the intersection of all keys, if needed)
    keys = list(state_dict.keys())
    keys.sort()  # Optional: sort keys to have a predictable order

    states = [state_dict[k] for k in keys]
    actions = [action_dict[k] for k in keys]
    ground_truths = [ground_truth_dict[k] for k in keys]

    # Load the mapping file and create a dictionary to map ground truth strings to numbers.
    mapping_file = os.path.join(args.demo, "mapping", "mapping.txt")
    mapping_dict = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    # parts[0] is the number and parts[1] is the ground truth label
                    mapping_dict[parts[1]] = int(parts[0])

    # Convert each ground truth list from strings to numbers using the mapping.
    for i in range(len(ground_truths)):
        ground_truths[i] = [mapping_dict[label] for label in ground_truths[i]]

    del state_files, action_files, ground_truth_files

    assert len(states) == len(actions) == len(ground_truths),\
        "Error: Mismatch in the number of state, action, and ground truth files."

    with open(f"{args.demo}/config.json") as f:
        config = json.load(f)
    max_episode_length = config['max_episode_length']

    # Pad the states, actions, and ground truths to the max episode length
    for i in range(len(states)):
        state_len = len(states[i])
        action_len = len(actions[i])
        truth_len = len(ground_truths[i])

        if state_len < max_episode_length:
            states[i] = np.pad(states[i], ((0, max_episode_length - state_len), (0, 0)), mode='edge')
        if action_len < max_episode_length:
            actions[i] = np.pad(actions[i], (0, max_episode_length - action_len), mode='edge')
        if truth_len < max_episode_length:
            # For ground truths, pad with the last number
            ground_truths[i].extend([ground_truths[i][-1]] * (max_episode_length - truth_len))

    states = np.array(states, dtype=np.float32)  # Change to float32
    actions = np.array(actions)

    train_test_split = np.random.permutation(len(states))

    train_states   = states[train_test_split[int(len(states)*args.test_size):]]
    train_actions  = actions[train_test_split[int(len(states)*args.test_size):]]

    test_states  = states[train_test_split[:int(len(states)*args.test_size)]]
    test_actions = actions[train_test_split[:int(len(states)*args.test_size)]]

    test_lengths = torch.tensor([len(state) for state in test_states], dtype=torch.long).to(device)
    test_inputs = (
        torch.tensor(test_states, dtype=torch.float32).to(device),
        torch.tensor(test_actions, dtype=torch.long).to(device)
    )

    all_data_states = torch.tensor(states, dtype=torch.float32).to(device)
    all_action_states = torch.tensor(actions, dtype=torch.long).to(device)

    return {
        'train': (train_states, train_actions),
        'test': (test_inputs, test_lengths),
        'all': (all_data_states, all_action_states, ground_truths)
    }