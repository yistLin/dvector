"""Dataset and DataLoader."""

import os
import pickle
from random import randint, sample

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Utterances(Dataset):
    """Utterances of a single speaker."""

    def __init__(self, data_dir, n_utterances=5, seg_len=128):
        """Args:
            data_dir (list): path to the directory of pickle files.
        """

        assert os.path.isdir(data_dir)

        self.data = []
        self.n_uttrances = n_utterances
        self.seg_len = seg_len

        for data_file in os.listdir(data_dir):
            data_path = os.path.join(data_dir, data_file)
            raw = pickle.load(open(data_path, "rb"))
            data = [torch.from_numpy(d) for d in raw if len(d) > seg_len]
            if len(data) < n_utterances:
                continue
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, sid):
        uttrs = sample(self.data[sid], self.n_uttrances)
        lefts = [randint(0, len(u) - self.seg_len) for u in uttrs]
        return [u[l:l+self.seg_len, :] for u, l in zip(uttrs, lefts)]


def pad_batch(batch):
    """Pad a whole batch of utterances."""
    flatten = [u for s in batch for u in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)
