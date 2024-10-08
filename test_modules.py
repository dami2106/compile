import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import utils


class TestILE(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim, max_num_segments,
                 temp_b=1., temp_z=1., latent_dist='gaussian', device='cuda'):
        super(TestILE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device


        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), stride=1),  # (2,50) -> (hidden_dim, 48)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output of the convolution
            # nn.Linear(3 * 3 * 16, state_dim)  # Fully co
        )

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


        self.action_embedding = nn.Embedding(action_dim, hidden_dim)

    def embed_input(self, inputs):

        batch_size, timesteps, c, h, w = inputs[0].shape
        inputs_reshaped = inputs[0].view(-1, c, h, w)  # Combine batch and time for conv (120, 1, 2, 50)

        conv_out = self.conv_layer(inputs_reshaped)  # Output shape: (120, hidden_dim, 1, 48)
        
        print(conv_out.shape)
    
        conv_out_flat = conv_out.view(batch_size * timesteps, -1)  # Flatten (120, hidden_dim * 24)

        print(conv_out_flat.shape)

        # state_embedding = self.state_embedding(conv_out_flat)  # Output shape: (120, hidden_dim)

        # # # Reshape back to (batch_size, timesteps, hidden_dim)
        # state_embedding = state_embedding.view(batch_size, timesteps, self.hidden_dim)
        # action_embedding = self.action_embedding(inputs[1])

        # embedding = torch.cat([state_embedding, action_embedding], dim=-1)
        # return embedding


    def forward(self, inputs, lengths):
        
        # Embed inputs.
        embeddings = self.embed_input(inputs)
