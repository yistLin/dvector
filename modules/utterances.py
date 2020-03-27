"""Dataset and DataLoader."""

import pickle
from random import randint, sample

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Utterances(Dataset):
    """Utterances of a single speaker."""

    def __init__(self, data_path, n_utterances=5, seg_len=128):
        """Args:
            data_path (list): path to pickle file.
        """
        with open(data_path, 'rb') as data_file:
            data = pickle.load(data_file)

        self.data = [[torch.from_numpy(u).cuda() for u in s] for s in data]
        self.seg_len = seg_len
        self.n_uttrances = n_utterances

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
