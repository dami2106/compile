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

from format_skills import determine_objectives, predict_clusters, create_KM_model, \
    get_latents, create_GMM_model, get_boundaries, calculate_metrics,get_skill_dict, print_skills_against_truth,\
        get_skill_accuracy, generate_elbow_plot, get_simple_obs_list, convert_dict_to_sota

from metrics import eval_mof, eval_f1, eval_miou, indep_eval_metrics, ClusteringMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=1000,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=12,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')


parser.add_argument('--log-interval', type=int, default=1,
                    help='Logging interval.')

parser.add_argument('--demo-file', type=str, default='trajectories/colours/15k',
                    help='path to the expert trajectories file')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=42,
                    help='Used to seed random number generators')
parser.add_argument('--results-file', type=str, default=None,
                    help='file where results are saved')
parser.add_argument('--train-model', action='store_true', 
                    help='Flag to indicate whether to train the model.')

parser.add_argument('--state-dim', type=int, default=11,
                    help='Size of the state dimension')
parser.add_argument('--action-dim', type=int, default=5,
                    help='Size of the action dimension')
parser.add_argument('--max-steps', type=int, default=12,
                    help='maximum number of steps in an expert trajectory')

parser.add_argument('--beta-b', type=float, default=0.1,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--beta-z', type=float, default=0.1,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--prior-rate', type=float, default=3.0,
                    help='maximum number of steps in an expert trajectory')

args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
run_ID = f"compile_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
if args.save_dir == '':
    run_dir = f"runs/{run_ID}"
else:
    run_dir = args.save_dir

if args.train_model:
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args), indent=4))
else:
    # print("Loaded Config File")
    config_file_path = os.path.join(run_dir, "config.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    args.train_model = False

data_path = args.demo_file
max_steps = args.max_steps

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

data_states = np.load(data_path + '_states.npy', allow_pickle=True)
data_actions = np.load(data_path + '_actions.npy', allow_pickle=True)

train_test_split = np.random.permutation(len(data_states))
train_test_split_ratio = 0.01

train_data_states = data_states[train_test_split[int(len(data_states)*train_test_split_ratio):]]
train_action_states = data_actions[train_test_split[int(len(data_states)*train_test_split_ratio):]]

test_data_states = data_states[train_test_split[:int(len(data_states)*train_test_split_ratio)]]
test_action_states = data_actions[train_test_split[:int(len(data_states)*train_test_split_ratio)]]

test_lengths = torch.tensor([max_steps-1] * len(test_data_states)).to(device)
test_inputs = (torch.tensor(test_data_states).to(device), torch.tensor(test_action_states).to(device))

perm = utils.PermManager(len(train_data_states), args.batch_size)


step = 0
rec = None
batch_loss = 0
batch_acc = 0
best_rec_acc = 0
best_nll = np.inf

if args.train_model:
    print('Training model with ', device)
    writer = SummaryWriter(log_dir = args.save_dir)
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
        loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args, beta_b=args.beta_b, beta_z=args.beta_z, prior_rate=args.prior_rate)

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
            print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(step, batch_loss, batch_acc))
            if batch_acc > best_rec_acc and batch_loss < best_nll:
                best_rec_acc = batch_acc
                best_nll = batch_loss
                model.save(os.path.join(run_dir, 'best_checkpoint.pth'))
            
            # Log to TensorBoard
            writer.add_scalar('Loss/nll_train', batch_loss, step)
            writer.add_scalar('Accuracy/rec_acc_eval', batch_acc, step)

        
        step += 1

    writer.close()
    model.save(os.path.join(run_dir, 'checkpoint.pth'))
    print("Best Reconstruction Accuracy: ", best_rec_acc)
    print("Best NLL: ", best_nll)
    if args.results_file:
        with open(args.results_file, 'a') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')
            f.write(' ')
            f.write(str(batch_acc))
            f.write(' ')
            f.write(str(model.K))
            f.write('\n')
else:
    print("Loading Model")
    model.load(os.path.join(run_dir, 'checkpoint.pth'))
    print("Model Loaded")


# print("Evaluating Model")
model.eval()


# Try load in the clustering model if it exists
try:
# print("Loading GMM Model")
    gmm_model = torch.load(os.path.join(run_dir, 'gmm_model.pth'))
    print("GMM Model Loaded")
except:
    print("Training Cluster Model")
    train_latents = get_latents(train_data_states, train_action_states, model, args, device)
    gmm_model = create_GMM_model(train_latents, args, args.num_segments)
    torch.save(gmm_model, os.path.join(run_dir, 'gmm_model.pth'))



all_true_boundaries = []
all_predicted_boundaries = []
dict_list_gmm = []

all_true_predicted_dicts = []

for i in range(len(test_data_states)):

    #Get a single datapoint from the test states
    single_input = (test_inputs[0][i].unsqueeze(0), test_inputs[1][i].unsqueeze(0))
    single_input_length = torch.tensor([single_input[0].shape[1]]).to(device)

    #Do a forward pass through the model using the single input point
    _, _, _, all_b, all_z = model.forward(single_input, single_input_length)

    #Get the predicted boundaries and the latents for each segment
    test_latents = [tensor.detach().cpu().numpy()[0].tolist() for tensor in all_z['samples']]
    predicted_boundaries =  [0] + [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]

    #Sort the predicted boundaries in ascending order (smallest to largest)
    predicted_boundaries = sorted(predicted_boundaries)

    #Skip incorrect segment predictions (when there is a boundary repeated)
    if len(set(predicted_boundaries)) < args.num_segments + 1:
        continue

    #Convert the input and action tensors to numpy arrays by detaching them from the GPU first
    state_array = single_input[0].cpu().detach().numpy()[0]
    action_array = single_input[1].cpu().detach().numpy()[0]

    # Uncomment this if you want to use a 5x5 obs
    # state_array = get_simple_obs_list(state_array)

    #Get a list of the true colour objectives at each time step and the true boundaries
    true_colours_each_timestep = determine_objectives(state_array)
    true_boundaries = get_boundaries(state_array)
    all_predicted_boundaries.append(predicted_boundaries)
    all_true_boundaries.append(true_boundaries)


    #Segment the states, actions, and colour objectives based on the predicted boundaries
    #Also save the segment indices
    state_segments = []
    action_segments = []
    # colour_objective_segments = []
    segment_indices = []
    for j in range(args.num_segments):
        start_idx = int(predicted_boundaries[j])
        end_idx = int(predicted_boundaries[j + 1]) 
        end_idx = end_idx if end_idx < len(state_array) - 1 else len(state_array) 

        state_segments.append(state_array[start_idx:end_idx])
        action_segments.append(action_array[start_idx:end_idx])
        # colour_objective_segments.append(true_colours_each_timestep[start_idx:end_idx])
        segment_indices.append((start_idx, end_idx))


    #Get the predicted clusters for each segment
    predict_colours_each_timestep = predict_clusters(gmm_model, test_latents)
    true_predicted_dict = get_skill_dict(state_array, state_segments, predict_colours_each_timestep)
    all_true_predicted_dicts.append(true_predicted_dict)
    # print("=============================================")
    # print("Predicted Skills:")
    # print(true_predicted_dict['Prediction'])
    # print("True Skills:")
    # print(true_predicted_dict['Truth'])
    # print("=============================================")

    # dict_list_gmm.append(pd.DataFrame( skill_dictionary ))




print("\n========== Segmentation Metrics: ==========")
overall_mse, overall_l2_distance, accuracy, precision, recall, f1_score = calculate_metrics(all_true_boundaries, all_predicted_boundaries)

# Format and align results for better readability
print(f"{'MSE:':<15} {overall_mse:.4f}")
print(f"{'L2 Distance:':<15} {overall_l2_distance:.4f}")
print(f"{'Accuracy:':<15} {accuracy:.4%}")
print(f"{'Precision:':<15} {precision:.4%}")
print(f"{'Recall:':<15} {recall:.4%}")
print(f"{'F1 Score:':<15} {f1_score:.4%}")
print("===========================================\n")

# SOTA Metrics calculations
torch_segs, np_segs, torch_truth, np_truth = convert_dict_to_sota(all_true_predicted_dicts)
mask = [torch.ones_like(seg).bool() for seg in torch_segs]

per_metrics = indep_eval_metrics(
    torch_segs, 
    torch_truth,
    mask,
    metrics=['mof', 'f1', 'miou']
)

mof_full, _ = eval_mof(
    np.concatenate(np_segs), 
    np.concatenate(np_truth),
    n_videos=len(np_segs)
)

f1_full, _ = eval_f1(
    np.concatenate(np_segs), 
    np.concatenate(np_truth),
    n_videos=len(np_segs)
)

miou_full, _ = eval_miou(
    np.concatenate(np_segs), 
    np.concatenate(np_truth),
    n_videos=len(np_segs)
)

print("\n=============== SOTA Metrics: ===============")
print(f"{'F1 Full:':<15} {f1_full:.4f}")
print(f"{'F1 Per:':<15} {per_metrics['f1']:.4f}")
print(f"{'MIOU Full:':<15} {miou_full:.4f}")
print(f"{'MIOU Per:':<15} {per_metrics['miou']:.4f}")
print(f"{'MOF Full:':<15} {mof_full:.4f}")
print(f"{'MOF Per:':<15} {per_metrics['mof']:.4f}")
print("==============================================\n")


# skill_acc_gmm = get_skill_accuracy(dict_list_gmm, args.num_segments)
# print("=============================================")
# print("Skill Accuracy:")
# # print(f"GMM: {skill_acc_gmm}")
# for acc in skill_acc_gmm:
#     print(f"{acc}")
# print()


