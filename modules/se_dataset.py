"""Dataset for speaker embedding."""

import os
import pickle
from random import randint, sample

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SEDataset(Dataset):
    """Sample utterances from a single speaker."""

    def __init__(self, data_dir, n_utterances=5, seg_len=128):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        assert os.path.isdir(data_dir)

        self.data_paths = []
        self.n_uttrances = n_utterances
        self.seg_len = seg_len

        for spkr_dir in os.listdir(data_dir):
            data_path = os.path.join(data_dir, spkr_dir)
            data = pickle.load(open(data_path, "rb"))
            if len(data) < n_utterances:
                continue
            self.data_paths.append(data_path)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, sid):
        data = pickle.load(open(self.data_paths[sid], "rb"))
        uttrs = sample(data, self.n_uttrances)
        lefts = [randint(0, len(uttr) - self.seg_len)
                 if len(uttr) > self.seg_len else None
                 for uttr in uttrs]
        sgmts = [uttr[left:left+self.seg_len, :]
                 if left is not None else uttr
                 for uttr, left in zip(uttrs, lefts)]
        return [torch.from_numpy(sgmt) for sgmt in sgmts]


def pad_batch(batch):
    """Pad a whole batch of utterances."""
    flatten = [u for s in batch for u in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)
