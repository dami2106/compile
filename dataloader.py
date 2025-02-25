import os
import glob
import numpy as np
import torch
import utils  # assuming you have a utils module that contains PermManager
from torch.nn.utils.rnn import pad_sequence

class EpisodeDatasetLoader:
    def __init__(self, data_path, num_actions, batch_size, device, test_split_ratio=0.01):
        """
        Initialize the loader with the directories for features, actions, and ground truth.
        
        Args:
            data_path (str): Root directory where the folders 'features', 'actions', and 'groundTruth' are located.
            num_actions (int): Number of action classes for one-hot encoding.
            batch_size (int): Batch size used for training (for the PermManager).
            device (torch.device): The device (cpu or cuda) on which tensors will be allocated.
            test_split_ratio (float): Fraction of episodes to be used as test set.
        """
        self.data_path = data_path
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.device = device
        self.test_split_ratio = test_split_ratio

        # Set up folder paths
        self.features_folder = os.path.join(data_path, 'features')
        self.actions_folder = os.path.join(data_path, 'actions')
        self.groundTruth_folder = os.path.join(data_path, 'groundTruth')

        # Lists to hold episodes (states, actions, ground truth)
        self.all_states = []
        self.all_actions = []
        self.all_ground_truth = []

        # Load episodes from files
        self._load_episodes()

        # Convert all episodes to torch tensors
        self._convert_to_tensors()

        # Split data into training and testing sets
        self._train_test_split()

        # Create a permutation manager for training batches
        self.perm = utils.PermManager(len(self.train_data_states), self.batch_size)

    def _load_episodes(self):
        """
        Load each episode from the features, actions, and groundTruth folders.
        Assumes filenames like "episode_0.npy", "episode_1.npy", etc.
        """
        episode_feature_files = sorted(glob.glob(os.path.join(self.features_folder, '*.npy')))
        for feature_file in episode_feature_files:
            # Extract base filename (e.g., "episode_0") to match corresponding files
            base_name = os.path.basename(feature_file).replace('.npy', '')
            action_file = os.path.join(self.actions_folder, base_name + '.npy')
            # Ground truth files might not have an extension; if they do, adjust accordingly.
            ground_truth_file = os.path.join(self.groundTruth_folder, base_name)
            # If your ground truth files have an extension, e.g. .npy, uncomment the next line:
            # ground_truth_file = os.path.join(self.groundTruth_folder, base_name + '.npy')
            
            # Load data from file
            states = np.load(feature_file, allow_pickle=True)
            actions = np.load(action_file, allow_pickle=True)
            ground_truth = np.load(ground_truth_file, allow_pickle=True)

            # Convert actions (a list/array of integers) into one-hot encoding
            actions_one_hot = np.eye(self.num_actions)[actions]

            self.all_states.append(states)
            self.all_actions.append(actions_one_hot)
            self.all_ground_truth.append(ground_truth)

    def _convert_to_tensors(self):
        """
        Convert the loaded episodes (which can be variable in length) into torch tensors,
        and move them to the specified device.
        """
        self.all_states = [torch.tensor(ep).to(self.device) for ep in self.all_states]
        self.all_actions = [torch.tensor(ep).to(self.device) for ep in self.all_actions]
        self.all_ground_truth = [torch.tensor(ep).to(self.device) for ep in self.all_ground_truth]

    def _train_test_split(self):
        """
        Split episodes into training and testing sets based on the test_split_ratio.
        """
        num_episodes = len(self.all_states)
        indices = np.random.permutation(num_episodes)
        split_index = int(num_episodes * self.test_split_ratio)

        test_indices = indices[:split_index]
        train_indices = indices[split_index:]

        self.train_data_states = [self.all_states[i] for i in train_indices]
        self.train_action_states = [self.all_actions[i] for i in train_indices]
        self.train_ground_truth = [self.all_ground_truth[i] for i in train_indices]

        self.test_data_states = [self.all_states[i] for i in test_indices]
        self.test_action_states = [self.all_actions[i] for i in test_indices]
        self.test_ground_truth = [self.all_ground_truth[i] for i in test_indices]

        # Optionally, bundle test inputs together.
        self.test_inputs = (self.test_data_states, self.test_action_states, self.test_ground_truth)

    def get_train_data(self):
        """
        Returns:
            Tuple containing training states, actions, and ground truth.
        """
        return self.train_data_states, self.train_action_states, self.train_ground_truth

    def get_test_data(self):
        """
        Returns:
            Tuple containing test states, actions, and ground truth.
        """
        return self.test_data_states, self.test_action_states, self.test_ground_truth

    def get_all_data(self):
        """
        Returns:
            Tuple containing all states, actions, and ground truth.
        """
        return self.all_states, self.all_actions, self.all_ground_truth

    def get_permutation_manager(self):
        """
        Returns:
            The permutation manager for batching training data.
        """
        return self.perm
    
    # Custom collate function to pad sequences.
    def episode_collate_fn(batch):
        """
        Expects a list of tuples: (states, actions, ground_truth).
        Pads each episode in the batch to the length of the longest episode.
        Returns padded tensors along with the original sequence lengths.
        """
        states, actions, ground_truth = zip(*batch)
        lengths = torch.tensor([s.shape[0] for s in states], dtype=torch.long)
        # Pad sequences along the time dimension (batch_first=True).
        states_padded = pad_sequence(states, batch_first=True)
        actions_padded = pad_sequence(actions, batch_first=True)
        ground_truth_padded = pad_sequence(ground_truth, batch_first=True)
        return states_padded, actions_padded, ground_truth_padded, lengths