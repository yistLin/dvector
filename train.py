#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train d-vector."""

import json
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from itertools import count
from multiprocessing import cpu_count
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import GE2EDataset, InfiniteDataLoader, collate_batch, infinite_iterator
from modules import AttentivePooledLSTMDvector, GE2ELoss


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
    comment,
):
    """Train a d-vector network."""

    # setup job name
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    job_name = f"{start_time}_{comment}" if comment is not None else start_time

    # setup checkpoint and log dirs
    checkpoints_path = Path(model_dir) / "checkpoints" / job_name
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(model_dir) / "logs" / job_name)

    # create data loader, iterator
    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    dataset = GE2EDataset(data_dir, metadata["speakers"], n_utterances, seg_len)
    trainset, validset = random_split(dataset, [len(dataset) - n_speakers, n_speakers])
    train_loader = InfiniteDataLoader(
        trainset,
        batch_size=n_speakers,
        num_workers=n_workers,
        collate_fn=collate_batch,
        drop_last=True,
    )
    valid_loader = InfiniteDataLoader(
        validset,
        batch_size=n_speakers,
        num_workers=n_workers,
        collate_fn=collate_batch,
        drop_last=True,
    )
    train_iter = infinite_iterator(train_loader)
    valid_iter = infinite_iterator(valid_loader)

    # display training infos
    assert len(trainset) >= n_speakers
    assert len(validset) >= n_speakers
    print(f"[INFO] Use {len(trainset)} speakers for training.")
    print(f"[INFO] Use {len(validset)} speakers for validation.")

    # build network and training tools
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dvector = AttentivePooledLSTMDvector(
        dim_input=metadata["n_mels"],
        seg_len=seg_len,
    ).to(device)
    dvector = torch.jit.script(dvector)
    criterion = GE2ELoss().to(device)
    optimizer = SGD(list(dvector.parameters()) + list(criterion.parameters()), lr=0.01)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.5)

    # record training infos
    pbar = tqdm(total=valid_every, ncols=0, desc="Train")
    running_train_loss, running_grad_norm = deque(maxlen=100), deque(maxlen=100)
    running_valid_loss = deque(maxlen=batch_per_valid)

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

        running_train_loss.append(loss.item())
        running_grad_norm.append(grad_norm.item())
        avg_train_loss = sum(running_train_loss) / len(running_train_loss)
        avg_grad_norm = sum(running_grad_norm) / len(running_grad_norm)

        pbar.update(1)
        pbar.set_postfix(loss=avg_train_loss, grad_norm=avg_grad_norm)

        if step % valid_every == 0:
            pbar.reset()

            for _ in range(batch_per_valid):
                batch = next(valid_iter).to(device)
                with torch.no_grad():
                    embd = dvector(batch).view(n_speakers, n_utterances, -1)
                    loss = criterion(embd)
                    running_valid_loss.append(loss.item())

            avg_valid_loss = sum(running_valid_loss) / len(running_valid_loss)

            tqdm.write(f"Valid: step={step}, loss={avg_valid_loss:.1f}")
            writer.add_scalar("Loss/train", avg_train_loss, step)
            writer.add_scalar("Loss/valid", avg_valid_loss, step)

        if step % save_every == 0:
            ckpt_path = checkpoints_path / f"dvector-step{step}.pt"
            dvector.cpu()
            dvector.save(str(ckpt_path))
            dvector.to(device)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("data_dir", type=str)
    PARSER.add_argument("model_dir", type=str)
    PARSER.add_argument("-n", "--n_speakers", type=int, default=64)
    PARSER.add_argument("-m", "--n_utterances", type=int, default=10)
    PARSER.add_argument("--seg_len", type=int, default=160)
    PARSER.add_argument("--save_every", type=int, default=10000)
    PARSER.add_argument("--valid_every", type=int, default=1000)
    PARSER.add_argument("--decay_every", type=int, default=100000)
    PARSER.add_argument("--batch_per_valid", type=int, default=100)
    PARSER.add_argument("--n_workers", type=int, default=cpu_count())
    PARSER.add_argument("--comment", type=str)
    train(**vars(PARSER.parse_args()))
