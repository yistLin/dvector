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

class SpeakerVerifier(nn.Module):
    """speaker verifier using d-vector"""

    def __init__(self, dvec_path=None, n_speaker=109):
        super(SpeakerVerifier, self).__init__()

        dvector_init = {
            'num_layers': 2,
            'dim_input': 512,
            'dim_cell': 256,
            'dim_emb': 128,
        }

        if dvec_path is not None:
            ckpt = torch.load(dvec_path)
            dvector_init.update(ckpt["dvector_init"])
            self.dvector = DVector(**dvector_init)
            self.dvector.load_state_dict(ckpt["state_dict"])
        else:
            self.dvector = DVector(**dvector_init)

        self.linear = nn.Linear(dvector_init["dim_emb"], n_speaker)

        for p in self.dvector.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """Forward data through network."""

        embeds = self.dvector(x)
        logits = self.linear(embeds)

        return logits
