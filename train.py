import argparse
import torch

import utils
import modules
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=4,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-symbols', type=int, default=5,
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num-segments', type=int, default=4,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=1,
                    help='Logging interval.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

model = modules.CompILE(
    input_dim=args.num_symbols + 1,  # +1 for EOS/Padding symbol.
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train model.
print('Training model...')
for step in range(args.iterations):
    data = None
    rec = None
    batch_loss = 0
    batch_acc = 0
    optimizer.zero_grad()

    # Generate data.
    data = []
    for _ in range(args.batch_size):
        data.append(utils.generate_toy_data(
            num_symbols=args.num_symbols,
            num_segments=args.num_segments))
    lengths = torch.tensor(list(map(len, data)))
    lengths = lengths.to(device)

    inputs = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    inputs = inputs.to(device)

    # Run forward pass.
    model.train()
    outputs = model.forward(inputs, lengths)
    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)

    loss.backward()
    optimizer.step()

    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        outputs = model.forward(inputs, lengths)
    
        acc, rec = utils.get_reconstruction_accuracy(inputs, outputs, args)

        # Accumulate metrics.
        batch_acc += acc.item()
        batch_loss += nll.item()
        print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(
            step, batch_loss, batch_acc))
        # print('input sample: {}'.format(inputs[-1, :lengths[-1] - 1]))
        # print('reconstruction: {}'.format(rec[-1]))


def get_cluster_model(model, example_count = 50):

    kmeans = KMeans(n_clusters=args.num_segments + 1, random_state=42, n_init='auto')
    all_latents = []

    for _ in range(example_count):
        specific_input = utils.generate_toy_data(
            num_symbols=args.num_symbols,
            num_segments=args.num_segments)
        lengths = torch.tensor([len(specific_input)])

        specific_input = specific_input.unsqueeze(0).to(device)  # Add batch dimension

        # Run the model on the specific input
        _, _, _, _, all_z = model.forward(specific_input, lengths)

        # print(specific_input)
        for t in all_z['samples']:
            tens = t.detach().numpy()[0].tolist()
            all_latents.append(tens)

    all_latents = np.array(all_latents)
    
    kmeans.fit(all_latents)

    return kmeans

#A function to predict a latent for an input sequence into a set of clusters
#Returns a list of clusters for each segment 
def predict_clusters(cluster_model, new_latents):
    # np_latents = [tensor.detach().numpy([0]) for tensor in new_latents]

    cluster_to_skill = {
        0 : 'A',
        1 : 'B',
        2 : 'C',
        3 : 'D',
        4 : 'E',
        5 : 'F',
        6 : 'G',
    }

    clusters = []
    for l in new_latents:
        cluster = cluster_model.predict([l])[0] + 1
        clusters.append(cluster_to_skill[cluster])
    return clusters

print('\nAnalysis of a given input on the trained model:')
model.eval()  # Switch to evaluation mode



k_mod = get_cluster_model(model, 50)


# # Select a specific input for analysis
# specific_input = utils.generate_toy_data(
#     num_symbols=args.num_symbols,
#     num_segments=args.num_segments)
# lengths = torch.tensor([len(specific_input)])
# specific_input = specific_input.unsqueeze(0).to(device)  # Add batch dimension

test_input = [1, 1, 1, 2, 2, 1, 1, 2, 2, 0]
specific_input = torch.tensor(test_input).unsqueeze(0).to(device)  # Add batch dimension
lengths = torch.tensor([len(test_input)]).to(device) 

# Run the model on the specific input
_, _, _, all_b, all_z = model.forward(specific_input, lengths)

# print(specific_input)
latents =  [tensor.detach().numpy()[0].tolist() for tensor in all_z['samples']]
boundary_positions = [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]
boundary_positions = [0] + boundary_positions 

input_array = specific_input.cpu().detach().numpy()[0]

segments = []
segment_indices = []
for i in range(len(boundary_positions) - 1):
    start_idx = int(boundary_positions[i])
    end_idx = int(boundary_positions[i + 1])
    segments.append(input_array[start_idx:end_idx])
    segment_indices.append((start_idx, end_idx))

print(input_array)
print(segments)
print(predict_clusters(k_mod, latents))
print()
