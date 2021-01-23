"""Dataset for speaker embedding."""

import random
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class GE2EDataset(Dataset):
    """Sample utterances from speakers."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        speaker_infos: dict,
        n_utterances: int,
        seg_len: int,
    ):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        self.data_dir = data_dir
        self.n_utterances = n_utterances
        self.seg_len = seg_len
        self.infos = []

        for uttr_infos in speaker_infos.values():
            feature_paths = [
                uttr_info["feature_path"]
                for uttr_info in uttr_infos
                if uttr_info["mel_len"] > seg_len
            ]
            if len(feature_paths) > n_utterances:
                self.infos.append(feature_paths)

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        feature_paths = random.sample(self.infos[index], self.n_utterances)
        uttrs = [
            torch.load(Path(self.data_dir, feature_path))
            for feature_path in feature_paths
        ]
        lefts = [random.randint(0, len(uttr) - self.seg_len) for uttr in uttrs]
        segments = [
            uttr[left : left + self.seg_len, :] for uttr, left in zip(uttrs, lefts)
        ]
        return segments


def collate_batch(batch):
    """Collate a whole batch of utterances."""
    flatten = [u for s in batch for u in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)
