#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize speaker embeddings."""

from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from warnings import filterwarnings

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from librosa.util import find_files
from tqdm import tqdm

from data import AudioToolkit


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("-c", "--checkpoint_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    return vars(parser.parse_args())


def path_to_mel(audio_path, speaker_name):
    """Path to wav, wav to mel."""
    wav = AudioToolkit.preprocess_wav(audio_path)
    mel = AudioToolkit.wav_to_logmel(wav)
    return speaker_name, mel


def visualize(data_dirs, checkpoint_path, output_path):
    """Visualize high-dimensional embeddings using t-SNE."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dvector = torch.jit.load(checkpoint_path).eval().to(device)

    print("[INFO] model loaded.")

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []

    n_spkrs = 0

    for data_dir in data_dirs:
        data_dir_path = Path(data_dir)
        for spkr_dir in [x for x in data_dir_path.iterdir() if x.is_dir()]:
            n_spkrs += 1
            audio_paths = find_files(spkr_dir)
            spkr_name = spkr_dir.name
            for audio_path in audio_paths:
                futures.append(executor.submit(path_to_mel, audio_path, spkr_name))

    mels, spkr_names = [], []

    for future in tqdm(futures, ncols=0, desc="Processing utterances"):
        spkr_name, mel = future.result()
        mels.append(mel)
        spkr_names.append(spkr_name)

    embs = []

    for mel in tqdm(mels, ncols=0, desc="Converting to embeddings"):
        mel_tensor = torch.FloatTensor(mel).to(device)
        with torch.no_grad():
            emb = dvector.embed_utterance(mel_tensor)
            emb = emb.detach().cpu().numpy()
        embs.append(emb)

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
    visualize(**parse_args())
