#!python
# -*- coding: utf-8 -*-
"""Walk through directories and extract files."""

import os


class DirsWalker:
    """Traverse through speaker directories."""

    def __init__(self, root_dir):
        assert os.path.isdir(root_dir)

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
