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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=50000,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=128,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Mini-batch size (for averaging gradients).')
parser.add_argument('--num-segments', type=int, default=4,
                    help='Number of segments in data generation.')
parser.add_argument('--save-dir', type=str, default='runs/wsws_static',
                    help='Directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=0,
                    help='Used to seed random number generators')
parser.add_argument('--feature-name', type=str, default='pca_features')
parser.add_argument('--data-dir', type=str, default='Data/wsws_static/wsws_static_pixels_big',
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
step = 0
for step in range(args.iterations):
    optimizer.zero_grad()
    # Sample a batch of data
    batch = perm.get_indices()
    batch_states = train_states[batch]
    batch_actions = train_actions[batch]
    lengths = torch.tensor([max_steps] * args.batch_size).to(device)
    inputs = (torch.from_numpy(batch_states).to(device), torch.from_numpy(batch_actions).to(device))

    # Forward pass
    model.train()
    outputs = model.forward(inputs, lengths)
    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
    loss.backward()
    optimizer.step()

    # Log training loss and components
    writer.add_scalar('Loss/total', loss.item(), step)
    writer.add_scalar('Loss/nll', nll.item(), step)
    writer.add_scalar('KL/z', kl_z.item(), step)
    writer.add_scalar('KL/b', kl_b.item(), step)

    if step % 5 == 0:
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model.forward(test_inputs, test_lengths)
            acc, _ = utils.get_reconstruction_accuracy(test_inputs, outputs, args)
        # Log evaluation metric
        writer.add_scalar('Accuracy/reconstruction', acc.item(), step)
        print(f'step: {step}, nll_train: {nll.item():.6f}, rec_acc_eval: {acc.item():.3f}')

# Save model checkpoint and close writer
model.save(os.path.join(run_dir, 'checkpoint.pth'))
writer.close()

model.eval()                        # turn off dropout, etc.
all_results = []                    # will hold outputs for each batch
all_latents = []

model.eval()
all_latents = []
all_boundaries = []

with torch.no_grad():
    N = all_states.shape[0]
    for idx in range(N):
        s = torch.from_numpy(all_states[idx:idx+1]).to(device)   # [1, T, state_dim]
        a = torch.from_numpy(all_actions[idx:idx+1]).to(device)  # [1, T, action_dim]

        lengths = torch.full(
            (1,), max_steps,
            dtype=torch.long,
            device=device
        )

        # forward → (_recs, _encs, _masks, _b, all_z)
        _, _, _, all_b, all_z = model((s, a), lengths)

        # all_z['logits'] is a list of length `segments`,
        # each element shape [1, 2*latent_dim] = [1, μ‖logσ²].
        mus = []
        for logits_z in all_z['logits']:
            # take the mean part μ = logits_z[:, :latent_dim]
            mu = logits_z[:, :args.latent_dim]
            # squeeze off the batch‐dim and to NumPy
            mus.append(mu.squeeze(0).cpu().numpy())

        boundary_onehots = all_b['samples']   # list of length max_num_segments

        # 4) convert one‐hots → actual index
        boundary_positions = [
            b.squeeze(0).argmax().item()
            for b in boundary_onehots
        ]

        all_boundaries.append(boundary_positions)
        all_latents.append(mus)

# stack into (n_eps, segments, latent_dim)
all_latents = np.stack(all_latents, axis=0)

# flatten to (n_eps*segments, latent_dim)
all_latents_flat = all_latents.reshape(-1, args.latent_dim)

print("Latents shape: ", all_latents.shape)
print("Latents flat shape: ", all_latents_flat.shape)

scaler = StandardScaler()
latents_scaled = scaler.fit_transform(all_latents_flat)

# 2. Choose your number of clusters (K)
K = 2   # ← pick based on your prior or use elbow/silhouette to pick

# 3. Fit K-means
km = KMeans(
    n_clusters=K,
    init='k-means++',   # smart centroid init
    n_init='auto',          # was 10
    max_iter=300,
    random_state=args.random_seed
).fit(latents_scaled)

# 4. Get assignments & centers
labels = km.labels_          # shape (n_eps * segments,)
centroids = km.cluster_centers_  # shape (K, latent_dim)

# reshape labels back into per-episode form
labels_per_episode = labels.reshape(N, -1)  # shape (N, segments)

# now build a per-episode list of (boundary_position, cluster_label)
episode_segment_info = []
for idx in range(N):
    boundaries = all_boundaries[idx]      # list of length `segments`
    seg_labels = labels_per_episode[idx]  # array of length `segments`
    # pair them up
    info = list(zip(boundaries, seg_labels.tolist()))
    # info[i] = (boundary_position_i, cluster_label_i)
    episode_segment_info.append(info)

# if you print for the first episode:
print("Episode 0 segments → (boundary_position, cluster_label):")
for i, (bpos, lbl) in enumerate(episode_segment_info[0]):
    print(f"  segment {i:2d}: boundary at t={bpos:3d}, label={lbl}")


T = all_states.shape[1]
n_segments = labels_per_episode.shape[1]

time_labels_per_episode = []

for idx in range(N):
    boundaries = all_boundaries[idx]          # e.g. [b0, b1, …, b_{S-1}]
    seg_labels = labels_per_episode[idx]      # e.g. [ℓ0, ℓ1, …, ℓ_{S-1}]
    assert len(boundaries) == len(seg_labels)

    # create an array of length T to hold your per‐timestep labels
    tl = np.empty(T, dtype=int)

    # for each segment i, paint label ℓi over [start, end)
    prev = 0
    for i, end in enumerate(boundaries):
        tl[prev:end] = seg_labels[i]
        prev = end
    # if the last boundary doesn’t exactly hit T, fill the remainder
    if prev < T:
        tl[prev:] = seg_labels[-1]

    time_labels_per_episode.append(tl)

# now, for example, print episode 0’s label‐timeline:
print("Episode 0 labels per timestep:")
print(time_labels_per_episode[0])  # shape (T,), each entry ∈ {0, …, K-1}
print(all_ground_truth[0])  # shape (T,), each entry ∈ {0, …, K-1}
