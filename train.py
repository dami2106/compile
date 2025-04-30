import argparse
import os
import sys
import datetime
import json

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
import modules

from dataloader import get_data

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from helpers import * 

from metrics import * 

from visualisation import * 

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=50000,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=256,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Mini-batch size (for averaging gradients).')
parser.add_argument('--num-segments', type=int, default=4,
                    help='Number of segments in data generation.')
parser.add_argument('--skills', type=int, default=2,
                    help='Number of skills in data generation.')
parser.add_argument('--save-dir', type=str, default='runs/wsws_static_symbolic',
                    help='Directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=0,
                    help='Used to seed random number generators')
parser.add_argument('--feature-name', type=str, default='symbolic_obs',)
parser.add_argument('--data-dir', type=str, default='Data/wsws_static/wsws_static_symbolic_big',
                    help='Directory where the data is stored')
args = parser.parse_args()

# Run configuration
run_ID = f"compile_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
run_dir = args.save_dir if args.save_dir else f"runs/{run_ID}"
os.makedirs(run_dir, exist_ok=True)
# Save config
with open(os.path.join(run_dir, "config.json"), "w") as f:
    f.write(json.dumps(vars(args), indent=4))
# TensorBoard writer
writer = SummaryWriter(log_dir=run_dir)
# Log config as text in TB
writer.add_text('config', json.dumps(vars(args), indent=2))

# Device & random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loading data
data_dict = get_data(device, args)
train_states = data_dict['train_states']
train_actions = data_dict['train_actions']
test_states = data_dict['test_states']
test_actions = data_dict['test_actions']
test_inputs = data_dict['test_inputs']
test_lengths = data_dict['test_lengths']
perm = data_dict['perm']
state_dim = data_dict['state_dim']
action_dim = data_dict['action_dim']
max_steps = data_dict['max_steps']

all_states = data_dict['all_states']
all_actions = data_dict['all_actions']
all_ground_truth = data_dict['all_ground_truth']

del data_dict

print("Data loaded")
print("Train states shape: ", train_states.shape)
print("Train actions shape: ", train_actions.shape)
print("Test states shape: ", test_states.shape)
print("Test actions shape: ", test_actions.shape)

# Model setup
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

# Training Loop
# step = 0
# for step in range(args.iterations):
#     optimizer.zero_grad()
#     # Sample a batch of data
#     batch = perm.get_indices()
#     batch_states = train_states[batch]
#     batch_actions = train_actions[batch]
#     lengths = torch.tensor([max_steps] * args.batch_size).to(device)
#     inputs = (torch.from_numpy(batch_states).to(device), torch.from_numpy(batch_actions).to(device))

#     # Forward pass
#     model.train()
#     outputs = model.forward(inputs, lengths)
#     loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
#     loss.backward()
#     optimizer.step()

#     # Log training loss and components
#     writer.add_scalar('Loss/total', loss.item(), step)
#     writer.add_scalar('Loss/nll', nll.item(), step)
#     writer.add_scalar('KL/z', kl_z.item(), step)
#     writer.add_scalar('KL/b', kl_b.item(), step)

#     if step % 5 == 0:
#         # Evaluation
#         model.eval()
#         with torch.no_grad():
#             outputs = model.forward(test_inputs, test_lengths)
#             acc, _ = utils.get_reconstruction_accuracy(test_inputs, outputs, args)
#         # Log evaluation metric
#         writer.add_scalar('Accuracy/reconstruction', acc.item(), step)
#         print(f'step: {step}, nll_train: {nll.item():.6f}, rec_acc_eval: {acc.item():.3f}')

# # Save model checkpoint and close writer
# model.save(os.path.join(run_dir, 'checkpoint.pth'))

model.load(os.path.join(run_dir, 'checkpoint.pth'))
writer.close()


model.eval()
all_latents = []
all_boundaries = []

# 1) extract latents & boundaries
all_latents, all_boundaries = extract_segment_latents(
    model, all_states, all_actions,
    max_steps, device, args.latent_dim
)

# 2) cluster segments
labels_flat, centroids = cluster_latents(
    all_latents, n_clusters=args.skills, random_seed=args.random_seed
)

# 3) split flat labels into per-episode lists
counts = [len(b) for b in all_boundaries]
labels_per_episode = np.split(labels_flat, np.cumsum(counts)[:-1])

# 4) get timestep-wise labels for each episode
# 1) load mapping

mapping_file = os.path.join(args.data_dir, 'mapping', 'mapping.txt')
with open(mapping_file, 'r') as f:
    lines = f.read().splitlines()
mapping_dict = {
    int(k): name
    for k, name in (
        line.split(maxsplit=1)
        for line in lines
        if line.strip()
    )
}

reverse_mapping = {v: k for k, v in mapping_dict.items()}

# 2) build preds, truths, mask on the fly
T = all_states.shape[1]
preds, truths = [], []
for b, lbls, gt in zip(all_boundaries, labels_per_episode, all_ground_truth):
    preds.append(labels_per_timestep(b, lbls, T))
    truths.append([reverse_mapping[x] for x in gt])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_pred = torch.tensor(preds, device=device)
all_gt   = torch.tensor(truths, device=device)
mask     = torch.ones_like(all_pred, dtype=torch.bool)

preds = np.array(preds)
truths = np.array(truths)

# 3) compute metrics
metrics = indep_eval_metrics(
    pred_labels_batch=[all_pred],
    gt_labels_batch=[all_gt],
    mask=[mask],
    metrics=['mof', 'f1', 'miou']
)

mof_full, _ = eval_mof(
    np.concatenate(preds), 
    np.concatenate(truths),
    n_videos=len(preds)
)
miou_full, _ = eval_miou(
    np.concatenate(preds), 
    np.concatenate(truths),
    n_videos=len(preds)
)
f1_full , _ = eval_f1(
    np.concatenate(preds), 
    np.concatenate(truths),
    n_videos=len(preds)
)

results = {
    'test_f1_full':  f1_full,
    'test_f1_per':   metrics['f1'],
    'test_miou_full':miou_full,
    'test_miou_per': metrics['miou'],
    'test_mof_full': mof_full,
    'test_mof_per':  metrics['mof'],
}


print("Results:")
print("----------------------------------")
for name, val in results.items():
    print(f"{name:<15}{val:.4f}")
print("----------------------------------")


B = all_pred.shape[0]
#Make a visualsation folder 
visualisation_dir = os.path.join(run_dir, 'visualisation')
os.makedirs(visualisation_dir, exist_ok=True)

for i in range(B):
    # grab the i-th episode
    pred_i = all_pred[i]   # torch.Tensor, shape [T]
    gt_i   = all_gt[i]     # torch.Tensor, shape [T]
    m_i    = mask[i]       # torch.BoolTensor, shape [T]

    # plot it
    fig = plot_segmentation_gt(gt_i, pred_i, m_i)
    
    #Sav it 
    fig.savefig(os.path.join(visualisation_dir, f"segmentation_{i}.png"), dpi=300)
    plt.close(fig)