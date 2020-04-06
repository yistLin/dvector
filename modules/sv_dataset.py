"""Dataset for speaker verification."""

import os
import pickle
import itertools
import json
from random import randint

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SVDataset(Dataset):
    """dataset for speaker verification"""

    def __init__(self, data_dir, seg_len, evaluate=False):
        """Args:
            data_dir (str): path to the directory of pickle files.
        """
        assert os.path.isdir(data_dir)

        self.seg_len = seg_len
        self.evaluate = evaluate
        self.spect = []
        self.label = []

        count = 0
        id_list = {}

        for data_file in sorted(os.listdir(data_dir)):
            data_path = os.path.join(data_dir, data_file)
            raw = pickle.load(open(data_path, 'rb'))
            data = [torch.from_numpy(d) for d in raw if len(d) > self.seg_len]
            if len(data) == 0:
                continue
            if not evaluate:
                label = [count] * len(data)
                id_list[count] = data_file[data_file.index(
                    '(') + 1:data_file.index(')')]
            else:
                label = [data_file] * len(data)
            self.spect.append(data)
            self.label.append(label)
            count += 1

        self.spect = list(itertools.chain(*self.spect))
        self.label = list(itertools.chain(*self.label))

        assert len(self.spect) == len(self.label)

        self.total = count

        if not evaluate:
            with open("speaker_info.json", "w") as f:
                json.dump(id_list, f)

    def __len__(self):
        return len(self.spect)

    def __getitem__(self, index):
        if not self.evaluate:
            start = randint(0, len(self.spect[index]) - self.seg_len)
            return self.spect[index][start:start + self.seg_len, :], self.label[index]
        else:
            return self.spect[index], self.label[index]


def pad_batch_with_label(batch):
    """Pad a whole batch of utterances with labels."""
    flatten = [s[0] for s in batch]
    label = [s[1] for s in batch]
    return pad_sequence(flatten, batch_first=True, padding_value=0), label
