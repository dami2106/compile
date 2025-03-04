import argparse
import torch 
import numpy as np
import modules 
import dataloader
import utils
from tqdm import tqdm

# --- ARGUMENT PARSING --- #
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=10,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=32,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=2,
                    help='Mini-batch size (for averaging gradients).')
parser.add_argument('--test-size', type=float, default=0.1,
                    help='Test split')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')
parser.add_argument('--action-dim', type=int, default=3,
                    help='Size of the action dimension')
parser.add_argument('--state-dim', type=int, default=3,
                    help='Size of the state dimension')

parser.add_argument('--beta-b', type=float, default=0.1,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--beta-z', type=float, default=0.1,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--prior-rate', type=float, default=3.0,
                    help='maximum number of steps in an expert trajectory')

parser.add_argument('--demo', type=str, default='Data',
                    help='path to the expert trajectories file')
parser.add_argument('--save', type=str, default='',
                    help='directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=42,
                    help='Used to seed random number generators')
parser.add_argument('--silent',  action='store_true',
                    help='Flag to indicate whether to print debugging information.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.random_seed) 
torch.manual_seed(args.random_seed)

model = modules.CompILE(
    state_dim=args.state_dim,
    action_dim=args.action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist,
    temp_b=args.beta_b,
    temp_z=args.beta_z,
    device=device).to(device)


parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], [])
optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)



all_data = dataloader.load_data(args, device)

train_states = all_data['train'][0]
train_actions = all_data['train'][1]

test_inputs = all_data['test'][0]
test_lengths = all_data['test'][1]

all_data_states = all_data['all'][0]
all_action_states = all_data['all'][1]
all_ground_truths = all_data['all'][2]

del all_data
perm = utils.PermManager(len(train_states), args.batch_size)

step = 0
rec = None
batch_loss = 0
batch_acc = 0

if not args.silent:
    progress_bar = tqdm(total=args.iterations, desc="Training", dynamic_ncols=True)

while step < args.iterations:
    optimizer.zero_grad()

    batch = perm.get_indices()
    batch_states, batch_actions = train_states[batch], train_actions[batch]
    lengths = torch.tensor([len(state) for state in batch_states]).to(device)
    inputs = (torch.tensor(batch_states).to(device), torch.tensor(batch_actions).to(device))

    model.train()
    outputs = model.forward(inputs, lengths)
    
    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
    loss.backward()
    optimizer.step()

    if step % 5 == 0 and not args.silent:
        # Run evaluation.
        model.eval()
        outputs = model.forward(test_inputs, test_lengths)
        acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

        batch_acc = acc.item()
        batch_loss = nll.item()

    if not args.silent: 
        progress_bar.set_postfix({
        "NLL": f"{batch_loss:.6f}",
        "Acc": f"{batch_acc:.3f}" if batch_acc is not None else "N/A"
        })
        progress_bar.update(1)

    step += 1 
