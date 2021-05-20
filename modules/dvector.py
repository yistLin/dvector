"""Build a model for d-vector speaker embedding."""

import abc
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DvectorInterface(nn.Module, metaclass=abc.ABCMeta):
    """d-vector interface."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            and hasattr(subclass, "seg_len")
            or NotImplemented
        )

    @abc.abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward a batch through network.

        Args:
            inputs: (batch, seg_len, mel_dim)

        Returns:
            embeds: (batch, emb_dim)
        """
        raise NotImplementedError

    @torch.jit.export
    def embed_utterance(self, utterance: Tensor) -> Tensor:
        """Embed an utterance by segmentation and averaging

        Args:
            utterance: (uttr_len, mel_dim) or (1, uttr_len, mel_dim)

        Returns:
            embed: (emb_dim)
        """
        assert utterance.ndim == 2 or (utterance.ndim == 3 and utterance.size(0) == 1)

        if utterance.ndim == 3:
            utterance = utterance.squeeze(0)

        if utterance.size(0) <= self.seg_len:
            embed = self.forward(utterance.unsqueeze(0)).squeeze(0)
        else:
            # Pad to multiple of hop length
            hop_len = self.seg_len // 2
            tgt_len = math.ceil(utterance.size(0) / hop_len) * hop_len
            zero_padding = torch.zeros(tgt_len - utterance.size(0), utterance.size(1))
            padded = torch.cat([utterance, zero_padding.to(utterance.device)])

            segments = padded.unfold(0, self.seg_len, self.seg_len // 2)
            segments = segments.transpose(1, 2)  # (batch, seg_len, mel_dim)
            embeds = self.forward(segments)
            embed = embeds.mean(dim=0)
            embed = embed.div(embed.norm(p=2, dim=-1, keepdim=True))

        return embed

    @torch.jit.export
    def embed_utterances(self, utterances: List[Tensor]) -> Tensor:
        """Embed utterances by averaging the embeddings of utterances

        Args:
            utterances: [(uttr_len, mel_dim), ...]

        Returns:
            embed: (emb_dim)
        """
        embeds = torch.stack([self.embed_utterance(uttr) for uttr in utterances])
        embed = embeds.mean(dim=0)
        return embed.div(embed.norm(p=2, dim=-1, keepdim=True))


class LSTMDvector(DvectorInterface):
    """LSTM-based d-vector."""

    def __init__(
        self,
        num_layers=3,
        dim_input=40,
        dim_cell=256,
        dim_emb=256,
        seg_len=160,
    ):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_cell, num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.seg_len = seg_len

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = self.embedding(lstm_outs[:, -1, :])  # (batch, dim_emb)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))  # (batch, dim_emb)


class AttentivePooledLSTMDvector(DvectorInterface):
    """LSTM-based d-vector with attentive pooling."""

    def __init__(
        self,
        num_layers=3,
        dim_input=40,
        dim_cell=256,
        dim_emb=256,
        seg_len=160,
    ):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_cell, num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.linear = nn.Linear(dim_emb, 1)
        self.seg_len = seg_len

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = torch.tanh(self.embedding(lstm_outs))  # (batch, seg_len, dim_emb)
        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))
