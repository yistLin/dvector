"""Build a model for d-vector speaker embedding."""

import torch
import torch.nn as nn


class DVector(nn.Module):
    """d-vector network"""

    def __init__(self, num_layers=3, dim_input=80, dim_cell=768, dim_emb=256):
        super(DVector, self).__init__()

        self.lstm = nn.LSTM(input_size=dim_input,
                            hidden_size=dim_cell,
                            num_layers=num_layers,
                            batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)

    def forward(self, x):
        """Forward data through network."""

        self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:, -1, :])
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        embeds_normalized = embeds.div(norm)

        return embeds_normalized
