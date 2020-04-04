"""Build a model for speaker verification."""

import torch
import torch.nn as nn

from .dvector import DVector


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
