import torch
import numpy as np

import utils
import modules
import os 

def pad_with_edge(lst, target_length):
    """
    Pads or truncates lst to exactly target_length, using the last element
    as the pad value (edge mode).
    """
    n = len(lst)
    if n >= target_length:
        return lst[:target_length]
    if n == 0:
        raise ValueError("Cannot pad an empty list in edge mode (no last element).")
    pad_token = lst[-1]
    # how many times to pad
    to_add = target_length - n
    return lst + [pad_token] * to_add

def load_asot_data(args):
    filenames = []
    for filename in os.listdir(args.data_dir + f'/{args.feature_name}'):
        true_name = filename.split('.')[0]
        filenames.append(true_name)


    data_files = []
    for filename in filenames:
        state_name = args.data_dir + f'/{args.feature_name}/' + filename + '.npy'
        action_name = args.data_dir + '/actions/' + filename + '.npy'
        ground_truth_name = args.data_dir + '/groundTruth/' + filename

        data_files.append((state_name, action_name, ground_truth_name))

    max_length = 0
    for file in data_files:
        #Load the npy file
        data = np.load(file[0])
        
        shape = data.shape #Length x feature_size 
        if shape[0] > max_length:
            max_length = shape[0]

    all_states = []
    all_actions = []
    all_ground_truth = []
    for file in data_files:
        state = np.load(file[0])
        action = np.load(file[1])
        with open(file[2], 'r') as f:
            ground_truth = f.read().splitlines()
        
        state = np.pad(state, ((0, max_length - state.shape[0]), (0, 0)), mode='edge')
        action = np.pad(action, (0, max_length - action.shape[0]), mode='edge')
        ground_truth = pad_with_edge(ground_truth, max_length)

        assert state.shape[0] == max_length
        assert action.shape[0] == max_length
        assert len(ground_truth) == max_length

        all_states.append(state)
        all_actions.append(action)
        all_ground_truth.append(ground_truth)



    all_states = np.array(all_states).astype(np.float32)
    all_actions = np.array(all_actions).astype(np.int64)


    del data_files, filenames

    return all_states, all_actions, all_ground_truth


def get_data(device, args):
    data_states, data_actions, data_truth = load_asot_data(args)


    state_dim = data_states.shape[2]
    action_dim = data_actions.max() + 1
    max_steps = data_states.shape[1]

    np.random.seed(args.random_seed) 
    train_test_split = np.random.permutation(len(data_states))
    train_test_split_ratio = 0.05

    train_states = data_states[train_test_split[int(len(data_states)*train_test_split_ratio):]]
    train_actions = data_actions[train_test_split[int(len(data_states)*train_test_split_ratio):]]

    test_states = data_states[train_test_split[:int(len(data_states)*train_test_split_ratio)]]
    test_actions = data_actions[train_test_split[:int(len(data_states)*train_test_split_ratio)]]

    test_lengths = torch.tensor([max_steps] * len(test_states)).to(device)
    test_inputs = (torch.tensor(test_states).to(device), torch.tensor(test_actions).to(device))

    perm = utils.PermManager(len(train_states), args.batch_size)

    return {
        'train_states': train_states,
        'train_actions': train_actions,
        'test_states': test_states,
        'test_actions': test_actions,
        'test_inputs': test_inputs,
        'test_lengths': test_lengths,
        'perm': perm,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_steps': max_steps,
        'all_states': data_states,
        'all_actions': data_actions,
        'all_ground_truth': data_truth
    }