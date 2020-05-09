#!python
# -*- coding: utf-8 -*-
"""Create perturbed utterances."""

import argparse
from os import listdir
from os.path import join as join_path

import torch
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from modules.dvector import DVector
from modules.audiotoolkit import AudioToolkit


def visualize(root_path, checkpoint_path, dvector_config_path,
              toolkit_config_path, result_path):
    """Visualize high-dimensional embeddings using t-SNE."""

    ckpt = torch.load(checkpoint_path)
    dvector = DVector.load_config_file(dvector_config_path).cuda()
    dvector.load_state_dict(ckpt["state_dict"])
    dvector.eval()
    audiotk = AudioToolkit.load_config_file(toolkit_config_path)

    uttrs = []
    sids = []

    spkr_ids = [entry for entry in listdir(root_path)]
    spkr_paths = [join_path(root_path, spkr) for spkr in spkr_ids]

    for spkr_id, spkr_path in zip(spkr_ids, spkr_paths):

        uttr_paths = librosa.util.find_files(spkr_path)

        with torch.no_grad():
            specs = [audiotk.file_to_mel_ndarray(u) for u in uttr_paths]

        uttrs += specs
        sids += [spkr_id] * len(specs)

    print("[INFO] utterances loaded.")

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
        palette=sns.color_palette(n_colors=len(spkr_ids)),
        data=data,
        legend="full"
    )
    plt.savefig(result_path)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str,
                        help="path to root directory of speaker directories")
    parser.add_argument("checkpoint_path", type=str,
                        help="path to saved model")
    parser.add_argument("dvector_config_path", type=str,
                        help="path to dvector configuration")
    parser.add_argument("toolkit_config_path", type=str,
                        help="path to toolkit configuration")
    parser.add_argument("result_path", type=str,
                        help="path to save plotted result")

    return parser.parse_args()


if __name__ == "__main__":
    visualize(**vars(parse_args()))
