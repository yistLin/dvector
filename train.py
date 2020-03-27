#!python
# -*- coding: utf-8 -*-
"""Train d-vector."""

import argparse
import pickle
import random

import tqdm
import torch
from torch.optim import SGD
from torch.utils.data import RandomSampler
from torch.nn.utils.rnn import pad_sequence

from tensorboardX import SummaryWriter

from modules.data import SpeakerUtterances
from modules.dvector import DVector
from modules.ge2e import GE2ELoss


def train(data_path, model_path,
          save_every, n_steps, n_speakers, n_utterances):
    """Train a d-vector network."""

    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file)

    assert n_speakers <= len(data)

    datasets = [SpeakerUtterances(datum) for datum in data]
    samplers = [RandomSampler(dts, replacement=True, num_samples=n_utterances)
                for dts in datasets]

    dvector = DVector(num_layers=3, dim_input=80,
                      dim_cell=256, dim_emb=128).cuda()
    optimizer = SGD(dvector.parameters(), lr=0.01)
    criterion = GE2ELoss(init_w=10.0, init_b=-5.0,
                         loss_method='softmax').cuda()
    writer = SummaryWriter()

    pbar = tqdm.trange(n_steps)
    for step in pbar:

        sids = random.sample(range(len(data)), k=n_speakers)
        samples = [datasets[sid][uid] for sid in sids for uid in samplers[sid]]
        padded = pad_sequence(samples, batch_first=True, padding_value=0)

        embd = dvector(padded).view(n_speakers, n_utterances, -1)

        loss = criterion(embd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss = {loss:.2f}")
        writer.add_scalar('GE2E Loss', loss, step)

        if (step + 1) % save_every == 0:
            torch.save(dvector.state_dict(), model_path)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str,
                        help="data path of utterances")
    parser.add_argument("model_path", type=str,
                        help="path to save model")
    parser.add_argument("-s", "--save_every", type=int, default=10000,
                        help="save model every [save_every] epochs")
    parser.add_argument("-i", "--n_steps", type=int, default=1000000,
                        help="# of steps")
    parser.add_argument("-n", "--n_speakers", type=int, default=32,
                        help="# of speakers per batch")
    parser.add_argument("-m", "--n_utterances", type=int, default=5,
                        help="# of utterances per speaker")

    return parser.parse_args()


if __name__ == "__main__":
    train(**vars(parse_args()))
