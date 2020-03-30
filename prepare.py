#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import os
import pickle
import argparse
from multiprocessing import Pool

from modules.dirswalker import DirsWalker
from modules.audioprocessor import AudioProcessor


def prepare(root_paths, save_dir, extensions, n_threads):
    """Extract audio files from directories and turn into spectrograms."""

    assert os.path.isdir(save_dir)

    n_speakers = 0
    n_utterances = 0
    extensions = extensions.split(",")

    for root_path in root_paths:

        walker = DirsWalker(root_path)

        for sid, spath in walker:

            paths = walker.utterances(extensions)

            print(f"Collecting {len(paths):4d} utterances from {spath}")

            n_speakers = n_speakers + (1 if len(paths) > 0 else 0)
            n_utterances = n_utterances + len(paths)

            if len(paths) == 0:
                continue

            save_path = os.path.join(save_dir, f"s{n_speakers:04d}({sid}).pkl")

            with Pool(n_threads) as pool:
                specs = pool.map(AudioProcessor.file2spectrogram, paths)

            with open(save_path, 'wb') as out_file:
                pickle.dump(specs, out_file)

    return n_speakers, n_utterances


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("root_paths", nargs='+',
                        help="root directory of directories of speakers")
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="path to the directory to save processed object")
    parser.add_argument("-e", "--extensions", type=str, default="wav",
                        help="file extensions to use")
    parser.add_argument("-t", "--n_threads", type=int, default=4,
                        help="# of threads to use")

    return parser.parse_args()


if __name__ == "__main__":
    N_SPEAKERS, N_UTTERANCES = prepare(**vars(parse_args()))
    print(f"{N_UTTERANCES} utterances of {N_SPEAKERS} speakers collected.")
