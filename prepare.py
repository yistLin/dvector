#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import argparse
import random
from os import listdir, makedirs
from os.path import basename, isdir, splitext, join as join_path

import torch
import librosa

from modules.audiotoolkit import AudioToolkit


def prepare(root_paths, save_dir, config_path, max_amount):
    """Extract audio files from directories and turn into spectrograms."""

    audiotk = AudioToolkit.load_config_file(config_path)

    n_speakers = 0

    for root_path in root_paths:

        spkr_ids = [entry for entry in listdir(root_path)
                    if isdir(join_path(root_path, entry))]
        spkr_paths = [join_path(root_path, spkr) for spkr in spkr_ids]

        for spkr_id, spkr_path in zip(spkr_ids, spkr_paths):

            uttr_paths = librosa.util.find_files(spkr_path)

            if max_amount is not None and len(uttr_paths) > max_amount:
                uttr_paths = random.choices(uttr_paths, k=max_amount)

            uttr_ids = [splitext(basename(u))[0] for u in uttr_paths]

            print(f"Collecting {len(uttr_paths)} utterances from {spkr_path}")

            if len(uttr_paths) == 0:
                continue

            n_speakers += 1

            specs_path = join_path(save_dir, f"s{n_speakers:04d}({spkr_id})")

            makedirs(specs_path)

            with torch.no_grad():
                specs = [audiotk.file_to_mel_tensor(u) for u in uttr_paths]

            for spec, uttr_id in zip(specs, uttr_ids):
                torch.save(spec, join_path(specs_path, uttr_id + '.pt'))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("root_paths", nargs='+',
                        help="root directory of directories of speakers")
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="path to the directory to save processed object")
    parser.add_argument("-c", "--config_path", type=str, required=True,
                        help="path to audio toolkit configuration")
    parser.add_argument("-m", "--max_amount", type=int, default=None,
                        help="maximum amount of utterances to be extracted")

    return parser.parse_args()


if __name__ == "__main__":
    prepare(**vars(parse_args()))
