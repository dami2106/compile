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
          get_skill_accuracy, get_simple_obs_list, get_simple_obs_list_from_layers

import test_modules


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=1000,
                    help='Number of training iterations.')

parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=32,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')


parser.add_argument('--demo-file', type=str, default='trajectories/colours/5k',
                    help='path to the expert trajectories file')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and config are saved')

parser.add_argument('--random-seed', type=int, default=42,
                    help='Used to seed random number generators')
parser.add_argument('--results-file', type=str, default=None,
                    help='file where results are saved')
parser.add_argument('--train-model', action='store_true', 
                    help='Flag to indicate whether to train the model.')


parser.add_argument('--action-dim', type=int, default=5,
                    help='Size of the action dimension')
parser.add_argument('--max-steps', type=int, default=12,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--out-channels', type=int, default=64,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--kernel', type=int, default=3,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--stride', type=int, default=1,
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
    print("Loaded Config File")
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



model = test_modules.TestILE(
    state_dim=(4, 5, 5),
    action_dim=args.action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    out_channels=args.out_channels,
    kernel_size=args.kernel,
    stride=1,
    latent_dist=args.latent_dist,
    device=device).to(device)

# parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], []) # test here
parameter_list = list(model.parameters())  # test here


optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)

# data_states = np.load(data_path + '_states.npy', allow_pickle=True).reshape(100, 12, 2, 50)
data_states = np.load(data_path + '_states.npy', allow_pickle=True) #Shzpe is 100, 12, 4, 5, 5
data_actions = np.load(data_path + '_actions.npy', allow_pickle=True)
 


# print(data_states.shape)
# print(data_actions.shape)


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

if args.train_model:
    # print('Training model with ', device)
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
        # print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(step, batch_loss, batch_acc))
        
        # Log to TensorBoard
        writer.add_scalar('Loss/nll_train', batch_loss, step)
        writer.add_scalar('Accuracy/rec_acc_eval', batch_acc, step)

        
        step += 1

    writer.close()
    model.save(os.path.join(run_dir, 'checkpoint.pth'))

else:
    # print("Loading Model")
    model.load(os.path.join(run_dir, 'checkpoint.pth'))
    # print("Model Loaded")


# print("Evaluating Model")
model.eval()


try:
    gmm_model = torch.load(os.path.join(run_dir, 'gmm_model.pth'), weights_only=False)
    # print("GMM Model Loaded")
except:
    # print("Training Cluster Model")
    train_latents = get_latents(train_data_states, train_action_states, model, args, device)
    gmm_model = create_GMM_model(train_latents, args, 3)
    torch.save(gmm_model, os.path.join(run_dir, 'gmm_model.pth'))



all_true_boundaries = []
all_predicted_boundaries = []
dict_list_gmm = []

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
    state_array = get_simple_obs_list_from_layers(single_input[0].cpu().detach().numpy()[0])
    action_array = single_input[1].cpu().detach().numpy()[0]

   


    #Get a list of the true colour objectives at each time step and the true boundaries
    true_colours_each_timestep = determine_objectives(state_array)
    true_boundaries = get_boundaries(state_array)
    all_predicted_boundaries.append(predicted_boundaries)
    all_true_boundaries.append(true_boundaries)


    #Segment the states, actions, and colour objectives based on the predicted boundaries
    #Also save the segment indices
    state_segments = []
    action_segments = []
    colour_objective_segments = []
    segment_indices = []
    for i in range(args.num_segments):
        start_idx = int(predicted_boundaries[i])
        end_idx = int(predicted_boundaries[i + 1]) 
        end_idx = end_idx if end_idx < len(state_array) - 1 else len(state_array) 

        state_segments.append(state_array[start_idx:end_idx])
        action_segments.append(action_array[start_idx:end_idx])
        colour_objective_segments.append(true_colours_each_timestep[start_idx:end_idx])
        segment_indices.append((start_idx, end_idx))


    #Get the predicted clusters for each segment
    clusters_gmm = predict_clusters(gmm_model, test_latents)
    

    try:
        skill_dictionary = get_skill_dict(state_array, state_segments, clusters_gmm)
        dict_list_gmm.append(pd.DataFrame( skill_dictionary ))
    except:
        print("Failed to get skill dictionary")
        print(state_array)
        print("----------------")
        print(state_segments)
        print("----------------")
        print(clusters_gmm)
        print("----------------")
        print(predicted_boundaries)
        print("----------------")
        print(true_boundaries)
        print("----------------")
    
    

    print_skills_against_truth(state_array, state_segments, clusters_gmm)



skill_acc_gmm = get_skill_accuracy(dict_list_gmm)
print("\n=============================================")
print("Segmentation Metrics:")
overall_mse, overall_l2_distance, accuracy, precision, recall, f1_score = calculate_metrics(all_true_boundaries, all_predicted_boundaries)
print(f"Overall MSE: {overall_mse}")
print(f"Overall L2 Distance: {overall_l2_distance}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")


print("=============================================")
print("Skill Accuracy:")
print(f"GMM: {skill_acc_gmm}")
for acc in skill_acc_gmm:
    print(f"{acc}")

# print(skill_acc_gmm[0][1], accuracy, overall_l2_distance)



