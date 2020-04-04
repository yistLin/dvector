#!python
# -*- coding: utf-8 -*-
"""Train speaker verifier with pre-trained d-vector."""

import argparse

import os
import random
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader

from tensorboardX import SummaryWriter

from modules.utterances import Utterances, SVDataset, pad_batch_with_label
from modules.dvector import DVector
from modules.verifier import SpeakerVerifier
from modules.ge2e import GE2ELoss


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="path to directory of processed utterances")
    parser.add_argument("model_dir", type=str,
                        help="path to directory for saving checkpoints")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None,
                        help="path to load saved checkpoint")
    parser.add_argument("-p", "--pretrained_dvector_path", type=str, default=None,
                        help="path to load pre-trained d-vector")
    parser.add_argument("-i", "--n_steps", type=int, default=1000000,
                        help="total # of steps")
    parser.add_argument("-s", "--save_every", type=int, default=2500,
                        help="save model every [save_every] epochs")
    parser.add_argument("-d", "--decay_every", type=int, default=100000,
                        help="decay learning rate every [decay_every] steps")
    parser.add_argument("-l", "--seg_len", type=int, default=128,
                        help="length of the segment of an utterance")
    parser.add_argument("-r", "--ratio", type=float, default=0.85,
                        help="ratio of data used to train verifier")

    return parser.parse_args()

def sample_index(index_range, p):
    return random.sample(range(index_range), k=int(index_range * p))


def train(data_dir, model_dir, checkpoint_path, pretrained_dvector_path,
          n_steps, save_every, decay_every, seg_len, ratio):
    """Train speaker verifier"""

    # setup
    total_steps = 0
    assert os.path.isdir(model_dir)

    #load data
    dataset = SVDataset(data_dir, seg_len)
    train_index = sample_index(len(dataset), ratio)
    valid_index = [x for x in range(len(dataset)) if x not in train_index]
    train_set = Subset(dataset, train_index)
    valid_set = Subset(dataset, valid_index)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True,
                        collate_fn=pad_batch_with_label, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=2, shuffle=False,
                        collate_fn=pad_batch_with_label, drop_last=False)
    train_loader_iter = iter(train_loader)
    print(f"Training starts with {train_set.dataset.total} speakers.")

    # load checkpoint
    ckpt = None
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path)
        dvector_path = ckpt["dvector_path"]

    # build network and training tools
    model = SpeakerVerifier(pretrained_dvector_path, dataset.total)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.5)
    if ckpt is not None:
        total_steps = ckpt["total_steps"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optmizier"])
        scheduler.load(ckpt["scheduler"])

    # prepare for traning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    writer = SummaryWriter(model_dir)
    pbar = tqdm.trange(n_steps)

    # start training
    for step in pbar:

        total_steps += 1

        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        data, label = batch
        logits = model((data.to(device)))

        loss = criterion(logits, torch.LongTensor(label).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"global = {total_steps}, loss = {loss:.4f}")
        writer.add_scalar("train_loss", loss, total_steps)

        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"ckpt-{total_steps}.tar")
            ckpt_dict = {
                "total_steps": total_steps,
                "dvector_path": dvector_path,
                "state_dict": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(ckpt_dict, ckpt_path)

        if (step + 1) % save_every == 0:
            val_acc = 0.0
            val_loss = 0.0
            for batch in valid_loader:
                data, label = batch
                with torch.no_grad():
                    logits = model(data.to(device))
                    pred = logits.argmax(dim=1)
                    val_acc += (pred == torch.LongTensor(label).to(device)).sum().item()
                    val_loss += criterion(logits, torch.LongTensor(label).to(device)).item()
            val_acc /= len(valid_set)
            val_loss /= len(valid_loader)
            writer.add_scalar("valid_accuracy", val_acc, total_steps)
            writer.add_scalar("valid_loss", val_loss, total_steps)

    print("Training completed.")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train(**vars(parse_args()))
