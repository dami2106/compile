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

from torch.nn.utils.rnn import pad_sequence

import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import utils
from modules import CompILE

import cnn_modules

from metrics import eval_mof, eval_f1, eval_miou, indep_eval_metrics, ClusteringMetrics


# ----------------- #
#  Argument Parser  #   
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=5,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=12,
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
                    help='directory where model and results etc are saved')

parser.add_argument('--random-seed', type=int, default=42,
                    help='Used to seed random number generators')
parser.add_argument('--train-model', action='store_true', 
                    help='Flag to indicate whether to train the model.')

parser.add_argument('--state-dim', type=int, default=3,
                    help='Size of the state dimension')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Size of the action dimension (range of actions)')


parser.add_argument('--out-channels', type=int, default=64,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--kernel', type=int, default=3,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--stride', type=int, default=1,
                    help='maximum number of steps in an expert trajectory')

parser.add_argument('--verbose',  action='store_true', default=False,
                    help='Flag to indicate whether to print debugging information.')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.random_seed) # there were some issue with reproducibility
torch.manual_seed(args.random_seed)

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


features_path = "Data/features"
actions_path = "Data/actions"
groundTruth_path = "Data/groundTruth"

def load_trajectories(features_path, actions_path):
    """Loads all trajectories and returns lists of tensors for states and actions."""
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
                
                #Ground truth file is a text file with eacu item on a new line 
                with open(ground_truth_file, 'r') as f:
                    truths = f.readlines()
                    truths = [truth.strip() for truth in truths]

                
                state_tensors.append(states)
                action_tensors.append(actions)
                ground_truth.append(truths)
    
    return state_tensors, action_tensors, ground_truth


all_states, all_actions, all_ground_truth = load_trajectories(features_path, actions_path)


train_test_split_ratio = 0.1
num_episodes = len(all_states)
indices = np.random.permutation(num_episodes)
split_idx = int(num_episodes * train_test_split_ratio)

train_indices, test_indices = indices[split_idx:], indices[:split_idx]

train_states = [all_states[i] for i in train_indices]
train_actions = [all_actions[i] for i in train_indices]
train_truth = [all_ground_truth[i] for i in train_indices]

test_states = [all_states[i] for i in test_indices]
test_actions = [all_actions[i] for i in test_indices]
test_truth = [all_ground_truth[i] for i in test_indices]

print(f"Number of training episodes: {len(train_states)}")
print(f"Number of testing episodes: {len(test_states)}")

# Pad sequences for batch processing
def pad_and_batch(data_list):
    return pad_sequence(data_list, batch_first=True, padding_value=0)

test_data_states = pad_and_batch(test_states)
test_action_states = pad_and_batch(test_actions)

test_inputs = (test_data_states.to(device), test_action_states.to(device))

perm = utils.PermManager(len(train_states), batch_size=32)
step = 0
batch_loss = 0
batch_acc = 0

writer = SummaryWriter(log_dir="runs/experiment1")

while step < 10000:  # Number of iterations
    optimizer.zero_grad()
    batch_indices = perm.get_indices()
    batch_states = [train_states[i] for i in batch_indices]
    batch_actions = [train_actions[i] for i in batch_indices]
    
    batch_states_padded = pad_and_batch(batch_states).to(device)
    batch_actions_padded = pad_and_batch(batch_actions).to(device)
    lengths = torch.tensor([len(seq) for seq in batch_states], dtype=torch.long).to(device)
    
    inputs = (batch_states_padded, batch_actions_padded)
    model.train()
    outputs = model.forward(inputs, lengths)

    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
    loss.backward()
    optimizer.step()
    
    # Evaluation
    model.eval()
    outputs = model.forward(test_inputs, torch.tensor([len(seq) for seq in test_states]).to(device))
    acc, _ = utils.get_reconstruction_accuracy(test_inputs, outputs, args)
    
    batch_acc = acc.item()
    batch_loss = nll.item()
    
    print(f'step: {step}, nll_train: {batch_loss:.6f}, rec_acc_eval: {batch_acc:.3f}')
    writer.add_scalar('Loss/nll_train', batch_loss, step)
    writer.add_scalar('Accuracy/rec_acc_eval', batch_acc, step)
    step += 1

# writer.close()
model.save("checkpoint.pth")
writer.close()
