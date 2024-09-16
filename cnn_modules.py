import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import utils


class CompILECNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_segments,
                 temp_b=1., temp_z=1., latent_dist='gaussian', action_dim=None):
        super(CompILECNN, self).__init__()

        self.input_dim = input_dim  # Assuming input_dim = 10 for 10x10 matrix
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_num_segments = max_num_segments
        self.temp_b = temp_b
        self.temp_z = temp_z
        self.latent_dist = latent_dist
        self.action_dim = action_dim  # Dimension of action ID embedding

        # CNN layers for input encoding (adjusted for 3 channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3 input channels (for 10x10x3)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened size after convolutions (assuming input 10x10)
        flattened_size = 64 * self.input_dim * self.input_dim

        # Trainable linear layer after CNN
        self.cnn_to_hidden = nn.Linear(flattened_size, hidden_dim)

        # Optional: embedding for action ID concatenation
        if action_dim is not None:
            self.action_embed = nn.Embedding(action_dim, hidden_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # LSTM for the recognition model
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        # Latent variable prediction (z)
        self.head_z_1 = nn.Linear(hidden_dim, hidden_dim)
        if latent_dist == 'gaussian':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim * 2)
        elif latent_dist == 'concrete':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim)
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        # Boundary prediction (b)
        self.head_b_1 = nn.Linear(hidden_dim, hidden_dim)
        self.head_b_2 = nn.Linear(hidden_dim, 1)

        # Decoder MLP for reconstruction
        self.decode_1 = nn.Linear(latent_dim, hidden_dim)
        self.decode_2 = nn.Linear(hidden_dim, input_dim)

    def masked_encode(self, inputs, mask):
        """Run masked CNN + RNN encoder on input sequence."""
        # Pass inputs through the CNN
        inputs = inputs.permute(0, 3, 1, 2)  # Re-arrange input to (batch, channels, height, width)
        cnn_output = self.cnn(inputs)
        hidden = self.cnn_to_hidden(cnn_output)

        # Layer normalization
        hidden = self.layer_norm(hidden)

        # Optional: concatenate action ID embedding if provided (for recognition model)
        if self.action_dim is not None:
            action_embedding = self.action_embed(torch.arange(inputs.size(0), device=inputs.device))
            hidden = torch.cat([hidden, action_embedding], dim=-1)

        # Initialize LSTM state
        lstm_state = utils.get_lstm_initial_state(inputs.size(0), self.hidden_dim, device=inputs.device)
        outputs = []

        for step in range(inputs.size(1)):
            lstm_state = self.lstm_cell(hidden, lstm_state)
            lstm_state = (mask[:, step, None] * lstm_state[0],
                          mask[:, step, None] * lstm_state[1])  # Apply mask
            outputs.append(lstm_state[0])

        return torch.stack(outputs, dim=1)

    # Reset LSTM state between trajectories
    def reset_lstm_state(self):
        self.lstm_state = None

    def forward(self, inputs, lengths):
        embeddings = self.embed(inputs)  # Assume embedding already exists
        mask = torch.ones(inputs.size(0), inputs.size(1), device=inputs.device)

        all_b = {'logits': [], 'samples': []}
        all_z = {'logits': [], 'samples': []}
        all_encs = []
        all_recs = []
        all_masks = []

        for seg_id in range(self.max_num_segments):
            encodings = self.masked_encode(embeddings, mask)
            all_encs.append(encodings)

            logits_b, sample_b = self.get_boundaries(encodings, seg_id, lengths)
            all_b['logits'].append(logits_b)
            all_b['samples'].append(sample_b)

            logits_z, sample_z = self.get_latents(encodings, sample_b)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)

            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)

            reconstructions = self.decode(sample_z, length=inputs.size(1))
            all_recs.append(reconstructions)

        return all_encs, all_recs, all_masks, all_b, all_z

    def save(self, path):
        checkpoint = {'model': self.state_dict()}
        for i, subpolicy in enumerate(self.subpolicies):
            checkpoint[f"subpolicy-{i}"] = subpolicy.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for i, subpolicy in enumerate(self.subpolicies):
            subpolicy.load_state_dict(checkpoint[f"subpolicy-{i}"])

    def play_from_observation(self, option, obs):
        with torch.no_grad():
            state = torch.tensor(obs).unsqueeze(0).unsqueeze(0).to(self.device).float()
            o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
            o_vector[0, option] = 1
            policy = self.decode(o_vector, state).cpu().numpy()
            termination = 0.
        return np.argmax(policy), termination

    def evaluate_score(self, states, actions):
        policies_probs = []
        with torch.no_grad():
            o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
            o_vector[0, 0] = 1
            policy = self.decode(o_vector, states)
            policy = policy.view(-1, policy.shape[-1]).cpu().numpy()
            max_probs = np.take_along_axis(policy, actions.view((-1, 1)).cpu().numpy(), 1).reshape(-1)
            policies_probs.append(max_probs)
            for option in range(1, self.latent_dim):
                o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
                o_vector[0, option] = 1
                policy = self.decode(o_vector, states)
                policy = policy.view(-1, policy.shape[-1]).cpu().numpy()
                prob = np.take_along_axis(policy, actions.view((-1, 1)).cpu().numpy(), 1).reshape(-1)
                policies_probs.append(prob)
                max_probs = np.maximum(max_probs, prob)
            policies_probs = np.array(policies_probs)
        return np.mean(max_probs), policies_probs
