"""Build a model for d-vector speaker embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DVector(nn.Module):
    """d-vector network"""

    def __init__(
        self, num_layers=3, dim_input=40, dim_cell=768, dim_emb=256, seg_len=160,
    ):
        super(DVector, self).__init__()

        self.lstm = nn.LSTM(dim_input, dim_cell, num_layers, batch_first=True)
        self.linear = nn.Linear(dim_cell, 1)
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.seg_len = seg_len

    def forward(self, inputs: Tensor):
        """Forward a batch through network.

        Args:
            inputs: (batch, seg_len, mel_dim)

        Returns:
            embeds: (batch, emb_dim)
        """
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, emb_dim)
        attn_weights = F.softmax(self.linear(lstm_outs).squeeze(-1), dim=-1).unsqueeze(
            -1
        )
        embeds = torch.sum(lstm_outs * attn_weights, dim=1)
        embeds = self.embedding(embeds)
        embeds = embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))
        return embeds

    @torch.jit.export
    def embed_utterance(self, utterance: Tensor):
        """Embed an utterance by segmentation and averaging

        Args:
            utterance: (uttr_len, mel_dim)

        Returns:
            embed: (emb_dim)
        """
        assert utterance.ndim == 2

        if utterance.size(1) <= self.seg_len:
            embed = self.forward(utterance.unsqueeze(0)).squeeze(0)
        else:
            segments = utterance.unfold(0, self.seg_len, self.seg_len // 2)
            embeds = self.forward(segments)
            embed = embeds.mean(dim=0)
            embed = embed.div(embed.norm(p=2, dim=-1, keepdim=True))

        return embed
