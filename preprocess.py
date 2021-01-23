#!/usr/bin/env python3
"""Preprocess script"""

import json
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
from typing import List
from uuid import uuid4
from warnings import filterwarnings

import torch
import torchaudio
from librosa.util import find_files
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Wav2Mel


class PreprocessDataset(torch.utils.data.Dataset):
    """Preprocess dataset."""

    def __init__(self, data_dirs: List[str], wav2mel):
        self.wav2mel = wav2mel
        self.speakers = set()
        self.infos = []

        for data_dir in data_dirs:
            speaker_dir_paths = [x for x in Path(data_dir).iterdir() if x.is_dir()]
            for speaker_dir_path in speaker_dir_paths:
                audio_paths = find_files(speaker_dir_path)
                speaker_name = speaker_dir_path.name
                self.speakers.add(speaker_name)
                for audio_path in audio_paths:
                    self.infos.append((speaker_name, audio_path))

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        speaker_name, audio_path = self.infos[index]
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        mel_tensor = self.wav2mel(wav_tensor, sample_rate)
        return speaker_name, mel_tensor


def preprocess(data_dirs, output_dir):
    """Preprocess audio files into features for training."""

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    wav2mel = Wav2Mel()
    wav2mel_jit = torch.jit.script(wav2mel)
    sox_effects_jit = torch.jit.script(wav2mel.sox_effects)
    log_melspectrogram_jit = torch.jit.script(wav2mel.log_melspectrogram)

    wav2mel_jit.save(str(output_dir_path / "wav2mel.pt"))
    sox_effects_jit.save(str(output_dir_path / "sox_effects.pt"))
    log_melspectrogram_jit.save(str(output_dir_path / "log_melspectrogram.pt"))

    dataset = PreprocessDataset(data_dirs, wav2mel_jit)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=cpu_count())

    infos = {
        "n_mels": wav2mel.n_mels,
        "speakers": {speaker_name: [] for speaker_name in dataset.speakers},
    }

    for speaker_name, mel_tensor in tqdm(dataloader, ncols=0, desc="Preprocess"):
        speaker_name = speaker_name[0]
        mel_tensor = mel_tensor.squeeze(0)
        random_file_path = output_dir_path / f"uttr-{uuid4().hex}.pt"
        torch.save(mel_tensor, random_file_path)
        infos["speakers"][speaker_name].append(
            {
                "feature_path": random_file_path.name,
                "mel_len": len(mel_tensor),
            }
        )

    with open(output_dir_path / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    filterwarnings("ignore")
    PARSER = ArgumentParser()
    PARSER.add_argument("data_dirs", type=str, nargs="+")
    PARSER.add_argument("-o", "--output_dir", type=str, required=True)
    preprocess(**vars(PARSER.parse_args()))
