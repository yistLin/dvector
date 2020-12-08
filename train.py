#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train d-vector."""

import json
from argparse import ArgumentParser
from itertools import count
from pathlib import Path
from multiprocessing import cpu_count
from datetime import datetime

import tqdm
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data import GE2EDataset, pad_batch
from modules import DVector, GE2ELoss


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model_dir", type=str)
    parser.add_argument("-n", "--n_speakers", type=int, default=64)
    parser.add_argument("-m", "--n_utterances", type=int, default=10)
    parser.add_argument("--seg_len", type=int, default=160)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--valid_every", type=int, default=1000)
    parser.add_argument("--decay_every", type=int, default=100000)
    parser.add_argument("--batch_per_valid", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    return vars(parser.parse_args())


def infinite_iterator(dataloader):
    """Infinitely yield a batch of data."""
    while True:
        for batch in iter(dataloader):
            yield batch


def train(
    data_dir,
    model_dir,
    n_speakers,
    n_utterances,
    seg_len,
    save_every,
    valid_every,
    decay_every,
    batch_per_valid,
    n_workers,
):
    """Train a d-vector network."""

    # read and log training infos
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoints_path = Path(model_dir) / "checkpoints" / start_time
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(model_dir) / "logs" / start_time)
    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # create data loader, iterator
    dataset = GE2EDataset(data_dir, metadata["speakers"], n_utterances, seg_len)
    trainset, validset = random_split(dataset, [len(dataset) - n_speakers, n_speakers])
    train_loader = DataLoader(
        trainset,
        batch_size=n_speakers,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=pad_batch,
        drop_last=True,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=n_speakers,
        num_workers=n_workers,
        collate_fn=pad_batch,
        drop_last=True,
    )
    train_iter = infinite_iterator(train_loader)
    valid_iter = infinite_iterator(valid_loader)

    assert len(trainset) >= n_speakers
    assert len(validset) >= n_speakers
    print(
        f"Training starts with {len(trainset)} speakers. "
        f"(and {len(validset)} speakers for validation)"
    )

    # build network and training tools
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dvector = DVector(dim_input=metadata["n_mels"], seg_len=seg_len,).to(device)
    dvector = torch.jit.script(dvector)
    criterion = GE2ELoss().to(device)
    optimizer = SGD(list(dvector.parameters()) + list(criterion.parameters()), lr=0.01)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.5)

    pbar = tqdm.tqdm(total=valid_every, ncols=0, desc="Train")
    train_losses, valid_losses = [], []

    # start training
    for step in count(start=1):

        batch = next(train_iter).to(device)
        embds = dvector(batch).view(n_speakers, n_utterances, -1)
        loss = criterion(embds)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(dvector.parameters()) + list(criterion.parameters()),
            max_norm=3,
            norm_type=2.0,
        )
        dvector.embedding.weight.grad *= 0.5
        dvector.embedding.bias.grad *= 0.5
        criterion.w.grad *= 0.01
        criterion.b.grad *= 0.01

        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        pbar.update(1)
        pbar.set_postfix(step=step, loss=loss.item(), grad_norm=grad_norm.item())

        if step % valid_every == 0:
            pbar.close()

            for _ in range(batch_per_valid):
                batch = next(valid_iter).to(device)

                with torch.no_grad():
                    embd = dvector(batch).view(n_speakers, n_utterances, -1)
                    loss = criterion(embd)
                    valid_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_valid_loss = sum(valid_losses) / len(valid_losses)
            print(f"Valid: loss={avg_valid_loss:.1f}")

            writer.add_scalar("Loss/train", avg_train_loss, step)
            writer.add_scalar("Loss/valid", avg_valid_loss, step)

            pbar = tqdm.tqdm(total=valid_every, ncols=0, desc="Train")
            train_losses, valid_losses = [], []

        if step % save_every == 0:
            ckpt_path = checkpoints_path / f"dvector-step{step}.pt"
            dvector.cpu()
            dvector.save(str(ckpt_path))
            dvector.to(device)


if __name__ == "__main__":
    train(**parse_args())
