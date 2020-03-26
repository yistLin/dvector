"""Dataset and DataLoader."""

import pickle
from functools import cached_property

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class VCTKDataset(Dataset):
    """VCTK dataset."""

    def __init__(self, pickle_path):
        """
        Args:
            pickle_path (string): Path to the pickle file.
        """
        self.file_path = pickle_path
        self.raw_data = self._load_data()
        self.data, self.idxs = self._tensorize_n_flatten_data()

    @cached_property
    def n_speakers(self):
        """Get the number of speakers."""
        return len(self.raw_data)

    @cached_property
    def n_utterances(self):
        """Get the number of utterances per speaker."""
        return len(self.raw_data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.idxs[idx]

    def _load_data(self):
        with open(self.file_path, 'rb') as data_file:
            raw_data = pickle.load(data_file)
        return raw_data

    def _tensorize_n_flatten_data(self):
        data = [uttr for spkr in self.raw_data for uttr in spkr]
        data = [torch.from_numpy(uttr).cuda() for uttr in data]

        idxs = torch.arange(self.n_speakers).unsqueeze(1)
        idxs = idxs.expand(self.n_speakers, self.n_utterances).flatten()

        lens = [len(uttr) for uttr in data]

        combined = list(zip(data, idxs, lens))
        combined = sorted(combined, key=lambda x: x[2])

        data, idxs, _ = zip(*combined)

        return list(data), list(idxs)


def pad_batch(batch):
    """Pad a batch of sequences."""

    data, idxs = zip(*batch)

    lens = [len(uttr) for uttr in data]
    data_pad = pad_sequence(data, batch_first=True, padding_value=0)

    return data_pad, list(idxs), lens
