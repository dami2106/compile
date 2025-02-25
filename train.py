import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tensorboardX import SummaryWriter
import utils  # assuming utils provides get_losses, get_reconstruction_accuracy, etc.


import argparse
import os
import sys
import datetime
import json
import torch
import numpy as np

import utils
import modules
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

# from format_skills import determine_objectives, predict_clusters, create_KM_model, \
#     get_latents, create_GMM_model, get_boundaries, calculate_metrics,get_skill_dict, print_skills_against_truth,\
#           get_skill_accuracy, get_simple_obs_list, get_simple_obs_list_from_layers, analyze_pickups,\
#               get_directional_dict, print_directions_against_truth, convert_dict_to_sota

import cnn_modules

from metrics import eval_mof, eval_f1, eval_miou, indep_eval_metrics, ClusteringMetrics


# ----------------- #
#  Argument Parser  #   
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=5,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=10,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=10,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=2,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')


parser.add_argument('--demo-file', type=str, default='Data',
                    help='path to the expert trajectories file')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and config are saved')

parser.add_argument('--random-seed', type=int, default=42,
                    help='Used to seed random number generators')
parser.add_argument('--results-file', type=str, default=None,
                    help='file where results are saved')
parser.add_argument('--train-model', action='store_true', 
                    help='Flag to indicate whether to train the model.')

parser.add_argument('--state-dim', type=int, default=3,
                    help='Size of the state dimension')
parser.add_argument('--action-dim', type=int, default=1,
                    help='Size of the action dimension')


parser.add_argument('--out-channels', type=int, default=64,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--kernel', type=int, default=3,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--stride', type=int, default=1,
                    help='maximum number of steps in an expert trajectory')

parser.add_argument('--verbose',  action='store_true', default=False,
                    help='Flag to indicate whether to print debugging information.')
args = parser.parse_args()

# ----------------- #
#  Initialization   #   

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# run_ID = f"compile_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
# if args.save_dir == '':
#     run_dir = f"runs/{run_ID}"
# else:
#     run_dir = args.save_dir

# if args.train_model:
#     os.makedirs(run_dir, exist_ok=True)

#     with open(os.path.join(run_dir, "config.json"), "w") as f:
#         f.write(json.dumps(vars(args), indent=4))
# else:
#     print("Loaded Config File")
#     config_file_path = os.path.join(run_dir, "config.json")
#     with open(config_file_path, "r") as f:
#         config = json.load(f)
#     args = argparse.Namespace(**config)
#     args.train_model = False

# data_path = args.demo_file
# max_steps = args.max_steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.random_seed) # there were some issue with reproducibility
torch.manual_seed(args.random_seed)

# ----------------- #

# model = test_modules.TestILE(
#     state_dim=(4, 5, 5),
#     action_dim=args.action_dim,
#     hidden_dim=args.hidden_dim,
#     latent_dim=args.latent_dim,
#     max_num_segments=args.num_segments,
#     out_channels=args.out_channels,
#     kernel_size=args.kernel,
#     stride=1,
#     latent_dist=args.latent_dist,
#     device=device).to(device)

# # parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], []) # test here
# parameter_list = list(model.parameters())  # test here

model = modules.CompILE(
    state_dim=args.state_dim,
    action_dim=args.action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist,
    device=device).to(device)


parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], [])

optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)

# Define a dataset that loads one episode per file.
class EpisodeDataset(Dataset):
    def __init__(self, features_dir, actions_dir, groundTruth_dir):
        # Get sorted lists of file paths so that matching episodes align.
        self.feature_files = sorted([os.path.join(features_dir, f)
                                     for f in os.listdir(features_dir)
                                     if f.endswith('.npy')])
        self.action_files = sorted([os.path.join(actions_dir, f)
                                    for f in os.listdir(actions_dir)
                                    if f.endswith('.npy')])
        self.gt_files = sorted([os.path.join(groundTruth_dir, f)
                                for f in os.listdir(groundTruth_dir)])
        # Ensure all folders have the same number of episodes.
        # assert len(self.feature_files) == len(self.action_files) == len(self.gt_files), \
        #     "Mismatch in number of episodes among features, actions, and ground truth."

        print(len(self.feature_files), len(self.action_files), len(self.gt_files))

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        # Load states, actions, and ground truth for one episode.
        states = np.load(self.feature_files[idx], allow_pickle=True)
        actions = np.load(self.action_files[idx], allow_pickle=True)
        
        #Load GT from text files 
        with open(self.gt_files[idx], 'r') as f:
            gt = f.readlines()
        gt = [x.strip() for x in gt]
        
        # Convert to tensors.
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        # gt = torch.tensor(gt, dtype=torch.str)
        return states, actions, gt

# Custom collate function that returns lists (no padding).
def collate_fn(batch):
    # Each item in the batch is a tuple: (states, actions, gt)
    states_list, actions_list, gt_list = zip(*batch)
    # Compute the lengths for each episode.
    lengths = [s.size(0) for s in states_list]
    # Return the lists as-is along with the lengths.
    return (list(states_list), list(actions_list), list(gt_list)), lengths

# Directories for the data.
features_dir = 'Data/features'
actions_dir = 'Data/actions'
groundTruth_dir = 'Data/groundTruth'

# Create the dataset.
dataset = EpisodeDataset(features_dir, actions_dir, groundTruth_dir)

# Shuffle and split the dataset into train and test sets (1% for testing).
indices = np.random.permutation(len(dataset))
test_ratio = 0.01
test_size = int(len(dataset) * test_ratio)
train_indices = indices[test_size:]
test_indices = indices[:test_size]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders using the custom collate function.
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

writer = SummaryWriter(log_dir=args.save_dir)
step = 0

while step < args.iterations:
    model.train()
    for batch_data, lengths in train_loader:
        # Unpack batch data (lists of tensors for states, actions, and ground truth).
        states_list, actions_list, gt_list = batch_data
        # Move each tensor to the target device.
        states_list = [s.to(device) for s in states_list]
        actions_list = [a.to(device) for a in actions_list]
        # Convert lengths to tensor if needed.
        lengths_tensor = torch.tensor(lengths).to(device)

        #Convert states and actions to tensors
        states_tensor = torch.tensor(states_list, dtype=torch.float)
        actions_tensor = torch.tensor(actions_list, dtype=torch.float)
        
        optimizer.zero_grad()
        # Prepare inputs for the model.
        inputs = (states_list, actions_list)
        outputs = model.forward(inputs, lengths_tensor)
        loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
        loss.backward()
        optimizer.step()
        
        # Run evaluation.
        model.eval()
        test_acc_list = []
        test_loss_list = []
        with torch.no_grad():
            for test_batch, test_lengths in test_loader:
                test_states_list, test_actions_list, test_gt_list = test_batch
                test_states_list = [s.to(device) for s in test_states_list]
                test_actions_list = [a.to(device) for a in test_actions_list]
                test_lengths_tensor = torch.tensor(test_lengths).to(device)
                test_inputs = (test_states_list, test_actions_list)
                test_outputs = model.forward(test_inputs, test_lengths_tensor)
                acc, rec = utils.get_reconstruction_accuracy(test_inputs, test_outputs, args)
                test_loss_list.append(nll.item())
                test_acc_list.append(acc.item())
        avg_test_acc = np.mean(test_acc_list)
        
        if args.verbose:
            print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(step, nll.item(), avg_test_acc))
        
        writer.add_scalar('Loss/nll_train', nll.item(), step)
        writer.add_scalar('Accuracy/rec_acc_eval', avg_test_acc, step)
        step += 1
        
        if step >= args.iterations:
            break

writer.close()
model.save(os.path.join(run_dir, 'checkpoint.pth'))