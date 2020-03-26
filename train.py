#!python
# -*- coding: utf-8 -*-
"""Train d-vector."""

import argparse

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange

from modules.data import VCTKDataset, pad_batch
from modules.dvector import DVector
from modules.ge2e import GE2ELoss


def train(data_path, model_path, save_every):
    """Train a d-vector network."""

    data = VCTKDataset(data_path)
    data_loader = DataLoader(data, batch_size=32, collate_fn=pad_batch)
    dvector = DVector(num_layers=3, dim_input=80,
                      dim_cell=768, dim_emb=256).cuda()
    optimizer = SGD(dvector.parameters(), lr=0.01)
    criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax')

    pbar = trange(1000)
    for epoch in pbar:

        loader_iter = iter(data_loader)
        embd_cnt = [0] * data.n_speakers
        embd_set = torch.empty(data.n_speakers, data.n_utterances, 256)

        for padded, sids, _ in loader_iter:

            print(padded.shape)

            embd = dvector(padded)

            for idx, sid in enumerate(sids):
                embd_set[sid, embd_cnt[sid], :] = embd[idx]
                embd_cnt[sid] += 1

        break

        loss = criterion(embd_set)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss = {loss:.4f}")

        if (epoch + 1) % save_every == 0:
            torch.save(dvector.state_dict(), model_path)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str,
                        help="data path of utterances")
    parser.add_argument("model_path", type=str,
                        help="path to save model")
    parser.add_argument("-s", "--save_every", type=int, default=1000,
                        help="save model every [save_every] epochs")

    return parser.parse_args()


if __name__ == "__main__":
    train(**vars(parse_args()))
