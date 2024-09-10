import argparse
import os
import sys
import datetime
import json
import torch
import numpy as np
# import gym

import utils
import modules

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=6,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='concrete',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=1,
                    help='Logging interval.')

parser.add_argument('--demo-file', type=str, default='trajectories/colours/',
                    help='path to the expert trajectories file')
parser.add_argument('--max-steps', type=int, default=11,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--atari-env-name', type=str, default='alien',
                    help='name of the atari env')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=0,
                    help='Used to seed random number generators')
parser.add_argument('--results-file', type=str, default=None,
                    help='file where results are saved')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
run_ID = f"compile_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
if args.save_dir == '':
    run_dir = f"runs/{run_ID}"
else:
    run_dir = args.save_dir
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "config.json"), "w") as f:
    f.write(json.dumps(vars(args), indent=4))

data_path = args.demo_file

state_dim = 11
action_dim = 4
max_steps = args.max_steps

device = torch.device('cuda' if args.cuda else 'cpu')
np.random.seed(args.random_seed) # there were some issue with reproducibility
torch.manual_seed(args.random_seed)

model = modules.CompILE(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist,
    device=device).to(device)

parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], [])

optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)

# model.load('checkpoint.pth')

data_states = np.load(data_path + 'states.npy', allow_pickle=True).astype('float32')
data_actions = np.load(data_path + 'actions.npy', allow_pickle=True)
data_actions = np.argmax(data_actions, axis=-1)
data_actions = np.hstack((data_actions, np.zeros((data_actions.shape[0], 1)))).astype('long')

train_test_split = np.random.permutation(len(data_states))
train_test_split_ratio = 0.01

train_data_states = data_states[train_test_split[int(len(data_states)*train_test_split_ratio):]]
train_action_states = data_actions[train_test_split[int(len(data_states)*train_test_split_ratio):]]

test_data_states = data_states[train_test_split[:int(len(data_states)*train_test_split_ratio)]]
test_action_states = data_actions[train_test_split[:int(len(data_states)*train_test_split_ratio)]]

test_lengths = torch.tensor([max_steps-1] * len(test_data_states)).to(device)
test_inputs = (torch.tensor(test_data_states).to(device), torch.tensor(test_action_states).to(device))

perm = utils.PermManager(len(train_data_states), args.batch_size)

# Train model.
print('Training model...')
# for step in range(args.iterations):
step = 0
rec = None
batch_loss = 0
batch_acc = 0
while step < args.iterations:
    optimizer.zero_grad()

    # Generate data.
    batch = perm.get_indices()
    batch_states, batch_actions = train_data_states[batch], train_action_states[batch]
    lengths = torch.tensor([max_steps] * args.batch_size).to(device)
    inputs = (torch.tensor(batch_states).to(device), torch.tensor(batch_actions).to(device))

    # Run forward pass.
    model.train()
    outputs = model.forward(inputs, lengths)
    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)

    loss.backward()
    optimizer.step()

    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        outputs = model.forward(test_inputs, test_lengths)
        acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

        # Accumulate metrics.
        batch_acc = acc.item()
        batch_loss = nll.item()
        print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(
            step, batch_loss, batch_acc))
        # print('input sample: {}'.format(test_inputs[1][-1, :test_lengths[-1] - 1]))
        # print('reconstruction: {}'.format(rec[-1]))
    
    step += 1


# Choose a single test input
single_test_input = test_data_states[0:1]  # Select the first trajectory for testing
single_test_action = test_action_states[0:1]  # Corresponding action sequence
single_test_length = torch.tensor([max_steps]).to(device)

# Convert to tensors and send to the appropriate device (CPU or GPU)
single_test_input_tensor = torch.tensor(single_test_input).to(device)
single_test_action_tensor = torch.tensor(single_test_action).to(device)
single_test_inputs = (single_test_input_tensor, single_test_action_tensor)

# Run forward pass on the single test input
model.eval()
all_encs, all_recs, all_masks, all_b, all_z = model.forward(single_test_inputs, single_test_length)


model.save(os.path.join(run_dir, 'checkpoint.pth'))


if args.results_file:
    with open(args.results_file, 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write(' ')
        f.write(str(batch_acc))
        f.write(' ')
        f.write(str(model.K))
        f.write('\n')
