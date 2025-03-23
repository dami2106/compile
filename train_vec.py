import argparse
import torch 
import numpy as np
import modules 
import dataloader
import utils
from tqdm import tqdm
import os 
from sklearn.mixture import GaussianMixture
from metrics import * 

# --- ARGUMENT PARSING --- #
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=10,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=8,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Mini-batch size (for averaging gradients).')
parser.add_argument('--test-size', type=float, default=0.1,
                    help='Test split')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')
parser.add_argument('--num-skills', type=int, default=3,
                    help='Number of skills in data generation.')
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
parser.add_argument('--random-seed', type=int, default=0,
                    help='Used to seed random number generators')
parser.add_argument('--silent',  action='store_true',
                    help='Flag to indicate whether to print debugging information.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.random_seed) 
torch.manual_seed(args.random_seed)
os.makedirs(args.save, exist_ok=True)

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

all_states = all_data['all'][0]
all_actions= all_data['all'][1]
all_truths= all_data['all'][2]

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


    if not args.silent:
        if step % 5 == 0:
            model.eval()
            outputs = model.forward(test_inputs, test_lengths)
            acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

            batch_acc = acc.item()
            batch_loss = nll.item()

        progress_bar.set_postfix({
        "NLL": f"{batch_loss:.6f}",
        "Acc": f"{batch_acc:.3f}" if batch_acc is not None else "N/A"
        })
        progress_bar.update(1)

    step += 1 

model.save(args.save + '/model.pth')
progress_bar.close() if not args.silent else None

# -- GMM MODEL -- #
all_latents = utils.get_latents(all_states, all_actions, model, device=device) #Latents shape: (num_segments, n_eps, latent_dim)
stacked_latents = all_latents.reshape(-1, all_latents.shape[-1])
gmm = GaussianMixture(n_components=args.num_skills, random_state=args.random_seed)
gmm.fit(stacked_latents)
torch.save(gmm, args.save + '/gmm.pth')
del all_latents, stacked_latents


model_skill_predictions = []
ground_truth_skills = []
for i, (state, action, truth) in enumerate(zip(all_states, all_actions, all_truths)):

    # Get a single datapoint and move it to the correct device
    state, action = state.unsqueeze(0), action.unsqueeze(0)
    single_input = (state, action)
    single_input_length = torch.tensor([state.shape[1]], device=device)

    # Forward pass through the model
    _, _, _, all_b, all_z = model.forward(single_input, single_input_length)

    # Extract predicted latents and boundaries
    test_latents = [z.detach().squeeze(0).cpu().numpy().tolist() for z in all_z['samples']]
    predicted_boundaries = [0] + [torch.argmax(b, dim=1).item() for b in all_b['samples']]

    # Ensure boundaries are unique and sorted
    predicted_boundaries = sorted(set(predicted_boundaries))

    # Skip incorrect segment predictions
    if len(predicted_boundaries) != args.num_segments + 1:
        print(f"Skipping episode {i} as the number of segments predicted is incorrect.")
        print(f"Predicted boundaries: {predicted_boundaries}")
        continue

    # Convert input tensors to numpy arrays
    state_array = state.squeeze(0).cpu().numpy()
    action_array = action.squeeze(0).cpu().numpy()
    latents = np.array(test_latents)

    predicted_state_classes = []
    for seg_idx in range(args.num_segments):
        start_idx = int(predicted_boundaries[seg_idx])
        end_idx = int(predicted_boundaries[seg_idx + 1]) 
        end_idx = end_idx if end_idx < len(state_array) - 1 else len(state_array) 

        segment_latent = latents[seg_idx]
        segment_class = gmm.predict(np.array(segment_latent).reshape(1, -1))[0]

        num_states_in_segment = end_idx - start_idx
        predicted_state_classes.extend([segment_class] * num_states_in_segment)

    model_skill_predictions.append(predicted_state_classes)
    ground_truth_skills.append(truth)

if len(model_skill_predictions) == 0:
    raise ValueError("No segments were correctly predicted. Please retrain the model.")
    

ground_truth_skills = np.array(ground_truth_skills)
model_skill_predictions = np.array(model_skill_predictions)

gt_labels_batch = torch.tensor(ground_truth_skills)
pred_labels_batch = torch.tensor(model_skill_predictions)
mask_batch = torch.ones_like(gt_labels_batch, dtype=torch.bool)


per_metrics = indep_eval_metrics(
    pred_labels_batch, 
    gt_labels_batch,
    mask_batch,
    metrics=['mof', 'f1', 'miou']
)

mof_full, _ = eval_mof(
    np.concatenate(model_skill_predictions), 
    np.concatenate(ground_truth_skills),
    n_videos=len(model_skill_predictions)
)

f1_full, _ = eval_f1(
    np.concatenate(model_skill_predictions), 
    np.concatenate(ground_truth_skills),
    n_videos=len(model_skill_predictions)
)

miou_full, _ = eval_miou(
    np.concatenate(model_skill_predictions), 
    np.concatenate(ground_truth_skills),
    n_videos=len(model_skill_predictions)
)

if not args.silent:
    print("\n=============== SOTA Metrics: ===============")
    print(f"{'F1 Full:':<15} {f1_full:.4f}")
    print(f"{'F1 Per:':<15} {per_metrics['f1']:.4f}")
    print(f"{'MIOU Full:':<15} {miou_full:.4f}")
    print(f"{'MIOU Per:':<15} {per_metrics['miou']:.4f}")
    print(f"{'MOF Full:':<15} {mof_full:.4f}")
    print(f"{'MOF Per:':<15} {per_metrics['mof']:.4f}")
    print("==============================================\n")