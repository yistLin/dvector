#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

from argparse import ArgumentParser
from multiprocessing import Pool
from os import listdir, makedirs
from os.path import basename, isdir, join as join_path, splitext

from numpy import save as save_npy
from librosa.util import find_files as find_audio_files

from modules.audioprocessor import AudioProcessor


def prepare(root_paths, save_dir, n_threads):
    """Extract audio files from directories and turn into spectrograms."""

    assert isdir(save_dir)

    n_speakers = 0

    for root_path in root_paths:

        spkr_ids = [entry for entry in listdir(root_path)
                    if isdir(join_path(root_path, entry))]
        spkr_paths = [join_path(root_path, spkr) for spkr in spkr_ids]

        for spkr_id, spkr_path in zip(spkr_ids, spkr_paths):

            uttr_paths = find_audio_files(spkr_path)
            uttr_ids = [splitext(basename(u))[0] for u in uttr_paths]

            print(f"Collecting {len(uttr_paths)} utterances from {spkr_path}")

            if len(uttr_paths) == 0:
                continue

            n_speakers += 1

            specs_path = join_path(save_dir, f"s{n_speakers:04d}({spkr_id})")

            makedirs(specs_path)

            with Pool(n_threads) as pool:
                specs = pool.map(AudioProcessor.file2spectrogram, uttr_paths)

            for spec, uttr_id in zip(specs, uttr_ids):
                save_npy(join_path(specs_path, uttr_id), spec)


def parse_args():
    """Parse command-line arguments."""

    parser = ArgumentParser()
    parser.add_argument("root_paths", nargs='+',
                        help="root directory of directories of speakers")
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="path to the directory to save processed object")
    parser.add_argument("-t", "--n_threads", type=int, default=4,
                        help="# of threads to use")

    return parser.parse_args()


if __name__ == "__main__":
    prepare(**vars(parse_args()))
