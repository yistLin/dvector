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
        self.infos = [
            [
                uttr_info
                for uttr_info in speaker_info["utterances"]
                if uttr_info["mel_len"] > seg_len
            ]
            for speaker_name, speaker_info in speaker_infos.items()
            if len(speaker_info["utterances"]) > n_utterances
        ]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        uttr_infos = random.sample(self.infos[index], self.n_utterances)
        uttrs = [
            torch.load(Path(self.data_dir, uttr_info["feature_path"]))
            for uttr_info in uttr_infos
        ]
        lefts = [random.randint(0, len(uttr) - self.seg_len) for uttr in uttrs]
        segments = [
            uttr[left : left + self.seg_len, :] for uttr, left in zip(uttrs, lefts)
        ]
        return segments


def pad_batch(batch):
    """Pad a whole batch of utterances."""
    flatten = [u for s in batch for u in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)
