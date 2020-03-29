#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import os
import pickle
import argparse
from multiprocessing import Pool

from modules.audioprocessor import AudioProcessor


class SpeakerDirsWalker:
    """Traverse through speaker directories."""

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.speaker_dirs = self.scan_root()
        self.speaker_idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_speaker()

    def scan_root(self):
        """Scan for speaker directories."""

        with os.scandir(self.root_dir) as entries:
            names = [entry.name for entry in entries if entry.is_dir()]

        return [(name, os.path.join(self.root_dir, name)) for name in names]

    def next_speaker(self):
        """Return next speaker directory path."""

        self.speaker_idx += 1

        if self.speaker_idx >= len(self.speaker_dirs):
            raise StopIteration()

        return self.speaker_dirs[self.speaker_idx]

    def utterances(self, exts):
        """Return list of path to utterances."""

        _, spath = self.speaker_dirs[self.speaker_idx]
        paths = [os.path.join(root, name)
                 for root, _, files in os.walk(spath) for name in files]
        filtered = list(filter(lambda x: x.split('.')[-1] in exts, paths))

        return filtered


def prepare(root_paths, save_dir, extensions):
    """Extract audio files from directories and turn into spectrograms."""

    assert os.path.isdir(save_dir)

    n_speakers = 0
    n_utterances = 0

    for root_path in root_paths:

        walker = SpeakerDirsWalker(root_path)

        for sid, spath in walker:

            paths = walker.utterances(extensions)

            print(f"Collecting {len(paths):4d} utterances from {spath}")

            n_speakers = n_speakers + (1 if len(paths) > 0 else 0)
            n_utterances = n_utterances + len(paths)

            if len(paths) == 0:
                continue

            save_path = os.path.join(save_dir, f"s{n_speakers:04d}({sid}).pkl")

            with Pool(6) as pool:
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

    return parser.parse_args()


if __name__ == "__main__":
    N_SPEAKERS, N_UTTERANCES = prepare(**vars(parse_args()))
    print(f"{N_UTTERANCES} utterances of {N_SPEAKERS} speakers collected.")
