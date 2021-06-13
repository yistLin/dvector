#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize speaker embeddings."""

from argparse import ArgumentParser
from pathlib import Path
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def equal_error_rate(test_dir, test_txt, wav2mel_path, checkpoint_path):
    """Compute equal error rate on test set."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir_path = Path(test_dir)
    test_txt_path = Path(test_txt)

    wav2mel = torch.jit.load(wav2mel_path)
    dvector = torch.jit.load(checkpoint_path).eval().to(device)

    pairs = []
    with test_txt_path.open() as file:
        for line in file:
            label, audio_path1, audio_path2 = line.strip().split()
            pairs.append((label, audio_path1, audio_path2))

    class MyDataset(Dataset):
        def __init__(self):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, index):
            label, path1, path2 = self.pairs[index]
            audio_path1 = test_dir_path / path1
            audio_path2 = test_dir_path / path2
            wav_tensor1, sample_rate = torchaudio.load(audio_path1)
            wav_tensor2, sample_rate = torchaudio.load(audio_path2)
            mel_tensor1 = wav2mel(wav_tensor1, sample_rate)
            mel_tensor2 = wav2mel(wav_tensor2, sample_rate)
            return int(label), mel_tensor1, mel_tensor2

    dataloader = DataLoader(
        MyDataset(),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=4,
    )

    labels, scores = [], []
    for label, mel1, mel2 in tqdm(dataloader, ncols=0, desc="Calculate Similarity"):
        mel1, mel2 = mel1.to(device), mel2.to(device)
        with torch.no_grad():
            emb1 = dvector.embed_utterance(mel1)
            emb2 = dvector.embed_utterance(mel2)
            score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        labels.append(label[0])
        scores.append(score.item())

    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    print("eer =", eer)
    print("thresh =", thresh)


if __name__ == "__main__":
    filterwarnings("ignore")
    PARSER = ArgumentParser()
    PARSER.add_argument("test_dir")
    PARSER.add_argument("test_txt")
    PARSER.add_argument("-w", "--wav2mel_path", required=True)
    PARSER.add_argument("-c", "--checkpoint_path", required=True)
    equal_error_rate(**vars(PARSER.parse_args()))
