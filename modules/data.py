"""Dataset and DataLoader."""

from random import randint

import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence


class SpeakerUtterances(Dataset):
    """Utterances of a single speaker."""

    def __init__(self, utterances, min_len=64, max_len=128):
        """
        Args:
            utterances (list): List of numpy.ndarray
        """
        self.raw_data = utterances
        self.data = self._to_cuda_tensor()
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uttr = self.data[idx]

        if len(uttr) <= self.max_len:
            return uttr

        l_bound = randint(0, len(uttr) - self.max_len)
        return uttr[l_bound:l_bound+self.max_len, :]

    def _to_cuda_tensor(self):
        return [torch.from_numpy(uttr).cuda() for uttr in self.raw_data]
