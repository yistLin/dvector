"""Dataset for speaker embedding."""

import os
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SEDataset(Dataset):
    """Sample utterances from speakers."""

    def __init__(self, data_dir, n_utterances=5, seg_len=128):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        self.n_utterances = n_utterances
        self.seg_len = seg_len

        self.spkr_dirs = [
            spkr_dir for spkr_dir in [os.path.join(data_dir, sid)
                                      for sid in os.listdir(data_dir)]
            if len(os.listdir(spkr_dir)) > n_utterances
        ]

    def __len__(self):

        return len(self.spkr_dirs)

    def __getitem__(self, sid):

        uttr_names = random.sample(os.listdir(self.spkr_dirs[sid]),
                                   self.n_utterances)

        uttrs = [torch.load(os.path.join(self.spkr_dirs[sid], uttr_name))
                 for uttr_name in uttr_names]

        lefts = [random.randint(0, len(uttr) - self.seg_len)
                 if len(uttr) > self.seg_len else None
                 for uttr in uttrs]

        sgmts = [uttr[left:left+self.seg_len, :]
                 if left is not None else uttr
                 for uttr, left in zip(uttrs, lefts)]

        return sgmts


def pad_batch(batch):
    """Pad a whole batch of utterances."""

    flatten = [u for s in batch for u in s]

    return pad_sequence(flatten, batch_first=True, padding_value=0)
