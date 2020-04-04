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

    for root_path in root_paths:

        walker = DirsWalker(root_path, extensions)

        for sdir in walker:

            paths = list(sdir)

            print(f"Collecting {len(paths):4d} utterances from {sdir.path}")

            if len(paths) == 0:
                continue

            n_speakers += 1

            with Pool(n_threads) as pool:
                specs = pool.map(AudioProcessor.file2spectrogram, paths)

            for spec, path in zip(specs, paths):
                save_path = os.path.join(save_dir, f"{path.replace('/', '_').replace('.wav', '')}.pkl")
                with open(save_path, 'wb') as out_file:
                    pickle.dump([spec], out_file)


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
    prepare(**vars(parse_args()))
