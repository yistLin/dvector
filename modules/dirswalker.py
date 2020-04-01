#!python
# -*- coding: utf-8 -*-
"""Walk through directories and extract files."""

import os


class DirsWalker:
    """Traverse through speaker directories."""

    def __init__(self, root_dir, exts):
        assert os.path.isdir(root_dir)

        with os.scandir(root_dir) as entries:
            names = [entry.name for entry in entries if entry.is_dir()]

        self.root_dir = root_dir
        self.exts = exts.split(",")

        self.name_dirs = [(n, os.path.join(root_dir, n)) for n in names]
        self.idx = -1

    def __len__(self):
        return len(self.name_dirs)

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1

        if self.idx >= len(self.name_dirs):
            raise StopIteration()

        return SubdirsWalker(*self.name_dirs[self.idx], self.exts)


class SubdirsWalker:
    """Traverse through utterances."""

    def __init__(self, dir_name, dir_path, exts):
        """Return list of path to utterances."""

        self.name = dir_name
        self.path = dir_path

        paths = [os.path.join(root, name)
                 for root, _, files in os.walk(dir_path) for name in files]

        self.uttr_paths = filter(lambda x: x.split('.')[-1] in exts, paths)

    def __iter__(self):
        return iter(self.uttr_paths)
