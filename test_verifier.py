#!python
# -*- coding: utf-8 -*-
"""Evaluate speaker verifier with pre-trained d-vector."""

import argparse

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.utterances import SVDataset, pad_batch_with_label
from modules.verifier import SpeakerVerifier


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="path to directory of processed utterances")
    parser.add_argument("-c", "--checkpoint_path", type=str, default="ver_models/ckpt-12500.tar",
                        help="path to load saved checkpoint")
    parser.add_argument("-l", "--seg_len", type=int, default=0,
                        help="length of the segment of an utterance")
    parser.add_argument("-s", "--speaker_dict", type=str, default="speaker_info.json",
                        help="path to load speaker ID dictionary")
    parser.add_argument("-o", "--output_path", type=str, default="predictions.json",
                        help="path to save the predictions")

    return parser.parse_args()

def test(data_dir, checkpoint_path, seg_len, speaker_dict, output_path):
    """Evaluate speaker verifier"""

    # load data
    dataset = SVDataset(data_dir, seg_len=seg_len, evaluate=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=pad_batch_with_label, drop_last=False)
    print(f"Evaluation starts with {len(dataset)} utterances.")

    # load checkpoint
    ckpt = torch.load(checkpoint_path)
    dvector_path = ckpt["dvector_path"]

    # load speaker info
    speaker_dict = json.load(open(speaker_dict))

    # build network and training tools
    model = SpeakerVerifier(dvector_path, len(speaker_dict))
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(ckpt["state_dict"])

    # prepare for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictions = []

    # start evaluation
    for batch in tqdm(loader):
        data, label = batch
        with torch.no_grad():
            logits = model(data.to(device))
        pred = logits.argmax(dim=1).flatten().tolist()
        for x, y in zip(pred, label):
            predictions.append((speaker_dict[str(x)], y))

    json.dump(predictions, open(output_path, 'w'))
    print("Evaluation completed.")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    test(**vars(parse_args()))
