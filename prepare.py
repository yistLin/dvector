#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import os
import pickle
import argparse

from tqdm import tqdm

from modules.audioprocessor import AudioProcessor


def prepare(root_path, save_path, n_speakers, n_utterances, min_frames):
    """Extract audio files from directories and turn them into spectrograms."""

    spkr_dirs = os.listdir(root_path)
    assert len(spkr_dirs) >= n_speakers

    spkr_list = []

    pbar = tqdm(total=n_speakers * n_utterances)

    for spkr in spkr_dirs:

        if len(spkr_list) == n_speakers:
            break

        uttr_list = []
        spkr_path = os.path.join(root_path, spkr)
        uttr_ents = os.listdir(spkr_path)
        uttr_wavs = list(filter(lambda x: x.endswith(".wav"), uttr_ents))

        if len(uttr_wavs) < n_utterances:
            continue

        for uttr in uttr_wavs:

            if len(uttr_list) == n_utterances:
                break

            uttr_path = os.path.join(spkr_path, uttr)
            uttr_spec = AudioProcessor.file2spectrogram(uttr_path)

            if len(uttr_spec) < min_frames:
                continue

            uttr_list.append(uttr_spec)

            pbar.update(1)

        spkr_list.append(uttr_list)

    pbar.close()

    assert len(spkr_list) == n_speakers

    with open(save_path, 'wb') as out_file:
        pickle.dump(spkr_list, out_file)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str,
                        help="root directory of directories of speakers")
    parser.add_argument("save_path", type=str,
                        help="path to save processed object")
    parser.add_argument("-n", "--n_speakers", default=10, type=int,
                        help="# of speakers")
    parser.add_argument("-m", "--n_utterances", default=10, type=int,
                        help="# of utterances per speaker")
    parser.add_argument("--min_frames", default=64, type=int,
                        help="minimum # of frames per utterance")

    return parser.parse_args()


if __name__ == "__main__":
    prepare(**vars(parse_args()))
