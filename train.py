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
import glob 

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

from dataloader import load_trajectories

# from metrics import eval_mof, eval_f1, eval_miou, indep_eval_metrics, ClusteringMetrics


# ----------------- #
#  Argument Parser  #   
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=500,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=64,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')


parser.add_argument('--demo-file', type=str, default='Data',
                    help='path to the expert trajectories file')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and results etc are saved')

parser.add_argument('--random-seed', type=int, default=0,
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


# --- Data Loading ---
# Get sorted lists of all state and action files.
state_files = sorted(glob.glob('Data/features' + '/*.npy'))
action_files = sorted(glob.glob('Data/actions' + '/*.npy'))

# Load each episode into a list.
data_states = [np.load(sf) for sf in state_files]
data_actions = [np.load(af) for af in action_files]

# Stack episodes into one array. (Assumes all episodes have the same length and observation dimensions)
data_states = np.stack(data_states)      # Shape: (num_episodes, episode_length, state_dim)
data_actions = np.stack(data_actions)      # Shape: (num_episodes, episode_length)

# --- Train/Test Split ---
num_episodes = data_states.shape[0]
indices = np.random.permutation(num_episodes)
train_test_split_ratio = 0.01
split_index = int(num_episodes * train_test_split_ratio)

test_indices = indices[:split_index]
train_indices = indices[split_index:]

train_data_states = data_states[train_indices]
train_action_states = data_actions[train_indices]
test_data_states = data_states[test_indices]
test_action_states = data_actions[test_indices]

# --- Pre-convert to Torch Tensors ---
# Convert states to float32 (to match model expectations) and actions to default integer type.
train_data_states = torch.tensor(train_data_states, dtype=torch.float32).to(device)
train_action_states = torch.tensor(train_action_states).to(device)
test_data_states = torch.tensor(test_data_states, dtype=torch.float32).to(device)
test_action_states = torch.tensor(test_action_states).to(device)

# Since all episodes have the same length.
episode_length = train_data_states.shape[1]
test_lengths = torch.tensor([episode_length] * test_data_states.shape[0]).to(device)

# Group inputs as tuples (states, actions).
train_inputs = (train_data_states, train_action_states)
test_inputs = (test_data_states, test_action_states)

# Optionally, if you need all data (for evaluation) you can do:
all_inputs = (torch.tensor(data_states, dtype=torch.float32).to(device),
              torch.tensor(data_actions).to(device))

# --- Training Loop Setup ---
# Using a permutation manager that shuffles indices for each epoch.
perm = utils.PermManager(len(train_data_states), args.batch_size)

step = 0
best_rec_acc = 0
best_nll = np.inf

if args.train_model:
    while step < args.iterations:
        optimizer.zero_grad()

        # Sample a batch of episodes.
        batch = perm.get_indices()
        # Directly index pre-converted tensors.
        batch_states = train_data_states[batch]
        batch_actions = train_action_states[batch]
        # Since all episodes are the same length, create a lengths tensor.
        lengths = torch.tensor([episode_length] * batch_states.shape[0]).to(device)
        inputs = (batch_states, batch_actions)

        # Run forward pass.
        model.train()
        outputs = model.forward(inputs, lengths)
        loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
        loss.backward()
        optimizer.step()

        # Run evaluation.
        model.eval()
        outputs = model.forward(test_inputs, test_lengths)
        acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

        # Accumulate metrics.
        batch_acc = acc.item()
        batch_loss = nll.item()

        if args.verbose:
            print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(step, batch_loss, batch_acc))
        
        # # Log to TensorBoard
        # writer.add_scalar('Loss/nll_train', batch_loss, step)
        # writer.add_scalar('Accuracy/rec_acc_eval', batch_acc, step)        
        step += 1

    # writer.add_scalar('Loss/nll_train', batch_loss, step)
    # writer.add_scalar('Accuracy/rec_acc_eval', batch_acc, step)
    step += 1

# writer.close()
# model.save("checkpoint.pth")
# writer.close()

# model.eval()

# for i in range(len(all_states)):

#     #Get a single datapoint from the test states
#     single_input = (all_inputs[0][i].unsqueeze(0), all_inputs[1][i].unsqueeze(0))
#     single_input_length = torch.tensor([single_input[0].shape[1]]).to(device)

#     #Do a forward pass through the model using the single input point
#     _, _, _, all_b, all_z = model.forward(single_input, single_input_length)

#     #Get the predicted boundaries and the latents for each segment
#     test_latents = [tensor.detach().cpu().numpy()[0].tolist() for tensor in all_z['samples']]
#     predicted_boundaries =  [0] + [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]

#     #Sort the predicted boundaries in ascending order (smallest to largest)
#     predicted_boundaries = sorted(predicted_boundaries)

#     # #Skip incorrect segment predictions (when there is a boundary repeated)
#     # if len(set(predicted_boundaries)) < args.num_segments + 1:
#     #     continue

#     #Convert the input and action tensors to numpy arrays by detaching them from the GPU first
#     single_raw_input = single_input[0].cpu().detach().numpy()[0]
#     action_array = single_input[1].cpu().detach().numpy()[0]

#     print()
#     print(predicted_boundaries)
#     print()
#     print(single_raw_input)

#     break
