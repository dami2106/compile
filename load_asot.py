import numpy as np
import os
import sys
import argparse

def load_asot_data(data_dir, args):
    filenames = []
    for filename in os.listdir(data_dir + '/pca_features'):
        true_name = filename.split('.')[0]
        filenames.append(true_name)


    data_files = []
    for filename in filenames:
        state_name = data_dir + f'/{args.feature_name}/' + filename + '.npy'
        action_name = data_dir + '/actions/' + filename + '.npy'
        ground_truth_name = data_dir + '/groundTruth/' + filename

        data_files.append((state_name, action_name, ground_truth_name))

    print(data_files)

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
        with open(file[2], 'r') as file:
            ground_truth = file.read().splitlines()


        if state.shape[0] < max_length:
            state = np.pad(state, ((0, max_length - state.shape[0]), (0, 0)), mode='edge')
            action = np.pad(action, (0, max_length - action.shape[0]), mode='edge')
            ground_truth = np.pad(ground_truth, ((0, max_length - len(ground_truth))), mode='edge')


        all_states.append(state)
        all_actions.append(action)
        all_ground_truth.append(ground_truth)

    all_states = np.array(all_states).astype(np.float32)
    all_actions = np.array(all_actions).astype(np.int64)

    del data_files, filenames

    return all_states, all_actions, all_ground_truth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--feature-name', type=str, default='pca_features')

    args = parser.parse_args()

    states, actions, truth = load_asot_data('Data/wsws_random/wsws_random_pixels', args)

    print("Final states shape", states.shape)
    print("Final actions shape", actions.shape)
    print("Final ground truth shape", len(truth), len(truth[0]))