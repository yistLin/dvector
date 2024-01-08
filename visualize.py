#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize speaker embeddings."""

from argparse import ArgumentParser
from pathlib import Path
from warnings import filterwarnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from librosa.util import find_files
from sklearn.manifold import TSNE
from tqdm import tqdm


def visualize(data_dirs, wav2mel_path, checkpoint_path, output_path):
    """Visualize high-dimensional embeddings using t-SNE."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2mel = torch.jit.load(wav2mel_path)
    dvector = torch.jit.load(checkpoint_path).eval().to(device)

    print("[INFO] model loaded.")

    n_spkrs = 0
    paths, spkr_names, mels = [], [], []

    for data_dir in data_dirs:
        data_dir_path = Path(data_dir)
        for spkr_dir in [x for x in data_dir_path.iterdir() if x.is_dir()]:
            n_spkrs += 1
            audio_paths = find_files(spkr_dir)
            spkr_name = spkr_dir.name
            for audio_path in audio_paths:
                paths.append(audio_path)
                spkr_names.append(spkr_name)

    for audio_path in tqdm(paths, ncols=0, desc="Preprocess"):
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        with torch.no_grad():
            mel_tensor = wav2mel(wav_tensor, sample_rate)
        mels.append(mel_tensor)

    embs = []

    for mel in tqdm(mels, ncols=0, desc="Embed"):
        with torch.no_grad():
            emb = dvector.embed_utterance(mel.to(device))
            emb = emb.detach().cpu().numpy()
        embs.append(emb)
        
    embs = np.array(embs)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    transformed = tsne.fit_transform(embs)

    print("[INFO] embeddings transformed.")

    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": spkr_names,
    }

    plt.figure()
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=sns.color_palette(n_colors=n_spkrs),
        data=data,
        legend="full",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    filterwarnings("ignore")
    PARSER = ArgumentParser()
    PARSER.add_argument("data_dirs", type=str, nargs="+")
    PARSER.add_argument("-w", "--wav2mel_path", required=True)
    PARSER.add_argument("-c", "--checkpoint_path", required=True)
    PARSER.add_argument("-o", "--output_path", required=True)
    visualize(**vars(PARSER.parse_args()))
