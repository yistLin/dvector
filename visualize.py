#!python
# -*- coding: utf-8 -*-
"""Create perturbed utterances."""

import argparse
from multiprocessing import Pool

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from modules.dirswalker import DirsWalker
from modules.dvector import DVector
from modules.audioprocessor import AudioProcessor


def visualize(root_path, result_path, checkpoint_path, extensions):
    """Visualize high-dimensional embeddings using t-SNE."""

    walker = DirsWalker(root_path)
    ckpt = torch.load(checkpoint_path)
    extensions = extensions.split(",")

    uttrs = []
    sids = []

    for sid, _ in walker:

        paths = walker.utterances(extensions)
        with Pool(4) as pool:
            specs = pool.map(AudioProcessor.file2spectrogram, paths)

        uttrs += specs
        sids += [sid] * len(specs)

    print("[INFO] utterances loaded.")

    dvector = DVector(**ckpt["dvector_init"]).cuda()
    dvector.load_state_dict(ckpt["state_dict"])
    dvector.eval()

    print("[INFO] model loaded.")

    embs = []

    for uttr in uttrs:
        uttr_tensor = torch.from_numpy(uttr).unsqueeze(0).cuda()
        emb = dvector(uttr_tensor).squeeze()
        emb = emb.detach().cpu().numpy()
        embs.append(emb)

    print("[INFO] utterances converted to embeddings.")

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    transformed = tsne.fit_transform(embs)

    print("[INFO] embeddings transformed.")

    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": sids,
    }

    plt.figure()
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=sns.color_palette(),
        data=data,
        legend="full"
    )
    plt.savefig(result_path)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str,
                        help="path to root directory of speaker directories")
    parser.add_argument("result_path", type=str,
                        help="path to save plotted result")
    parser.add_argument("-d", "--checkpoint_path", type=str,
                        help="path to saved model")
    parser.add_argument("-e", "--extensions", type=str, default="wav,flac",
                        help="extensions of files to be extracted")

    return parser.parse_args()


if __name__ == "__main__":
    visualize(**vars(parse_args()))
