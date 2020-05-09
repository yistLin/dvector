#!python
# -*- coding: utf-8 -*-
"""Train d-vector."""

import argparse

import os
import tqdm
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from tensorboardX import SummaryWriter

from modules.se_dataset import SEDataset, pad_batch
from modules.dvector import DVector
from modules.ge2e import GE2ELoss


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str,
                        help="path to directory of training data.")
    parser.add_argument("model_dir", type=str,
                        help="path to directory for saving checkpoints")
    parser.add_argument("config_path", type=str,
                        help="path to configuration of dvector")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None,
                        help="path to load saved checkpoint")
    parser.add_argument("-i", "--n_steps", type=int, default=1000000,
                        help="total # of steps")
    parser.add_argument("-s", "--save_every", type=int, default=10000,
                        help="save model every [save_every] steps")
    parser.add_argument("-t", "--test_every", type=int, default=1000,
                        help="test on validation set every [test_every] steps")
    parser.add_argument("-d", "--decay_every", type=int, default=100000,
                        help="decay learning rate every [decay_every] steps")
    parser.add_argument("-n", "--n_speakers", type=int, default=32,
                        help="# of speakers per batch")
    parser.add_argument("-m", "--n_utterances", type=int, default=5,
                        help="# of utterances per speaker")
    parser.add_argument("-l", "--seg_len", type=int, default=128,
                        help="length of the segment of an utterance")

    return parser.parse_args()


def train(train_dir, model_dir, config_path, checkpoint_path,
          n_steps, save_every, test_every, decay_every,
          n_speakers, n_utterances, seg_len):
    """Train a d-vector network."""

    # setup
    total_steps = 0

    # load data
    dataset = SEDataset(train_dir, n_utterances, seg_len)
    train_set, valid_set = random_split(dataset, [len(dataset)-2*n_speakers,
                                                  2*n_speakers])
    train_loader = DataLoader(train_set, batch_size=n_speakers,
                              shuffle=True, num_workers=4,
                              collate_fn=pad_batch, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_speakers,
                              shuffle=True, num_workers=4,
                              collate_fn=pad_batch, drop_last=True)
    train_iter = iter(train_loader)

    assert len(train_set) >= n_speakers
    assert len(valid_set) >= n_speakers
    print(f"Training starts with {len(train_set)} speakers. "
          f"(and {len(valid_set)} speakers for validation)")

    # build network and training tools
    dvector = DVector().load_config_file(config_path)
    criterion = GE2ELoss()
    optimizer = SGD(dvector.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.5)

    # load checkpoint
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path)
        total_steps = ckpt["total_steps"]
        dvector.load_state_dict(ckpt["state_dict"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    # prepare for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dvector = dvector.to(device)
    criterion = criterion.to(device)
    writer = SummaryWriter(model_dir)
    pbar = tqdm.trange(n_steps)

    # start training
    for step in pbar:

        total_steps += 1

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        embd = dvector(batch.to(device)).view(n_speakers, n_utterances, -1)

        loss = criterion(embd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"global = {total_steps}, loss = {loss:.4f}")
        writer.add_scalar("training Loss", loss, total_steps)

        if (step + 1) % test_every == 0:
            batch = next(iter(valid_loader))
            embd = dvector(batch.to(device)).view(n_speakers, n_utterances, -1)
            loss = criterion(embd)
            writer.add_scalar("validation loss", loss, total_steps)

        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"ckpt-{total_steps}.tar")
            ckpt_dict = {
                "total_steps": total_steps,
                "state_dict": dvector.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(ckpt_dict, ckpt_path)

    print("Training completed.")


if __name__ == "__main__":
    train(**vars(parse_args()))
