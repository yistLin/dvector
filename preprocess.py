#!/usr/bin/env python3
"""Preprocess script"""

import json
from pathlib import Path
from uuid import uuid4
from argparse import ArgumentParser
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import torch
from tqdm import tqdm
from librosa.util import find_files

from data import AudioToolkit


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    return vars(parser.parse_args())


def load_process_save(audio_path, output_dir_path, speaker_name):
    """Load an audio file, process, and save object."""

    wav = AudioToolkit.preprocess_wav(audio_path)

    if len(wav) < 10:
        return speaker_name, None

    mel = AudioToolkit.wav_to_logmel(wav)
    mel_tensor = torch.FloatTensor(mel)

    random_file_path = output_dir_path / f"uttr-{uuid4().hex}.pt"
    torch.save(mel_tensor, random_file_path)

    return (
        speaker_name,
        {
            "feature_path": random_file_path.name,
            "audio_path": audio_path,
            "mel_len": len(mel),
        },
    )


def main(data_dirs, output_dir, n_workers):
    """Preprocess audio files into features for training."""

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=n_workers)
    futures = []

    infos = {
        "n_mels": AudioToolkit.n_mels,
        "speakers": {},
    }

    for data_dir_path in data_dirs:
        data_dir = Path(data_dir_path)
        speaker_dirs = [x for x in data_dir.iterdir() if x.is_dir()]

        for speaker_dir in speaker_dirs:
            audio_paths = find_files(speaker_dir)

            speaker_name = speaker_dir.name
            infos["speakers"][speaker_name] = {
                "speaker_dir_path": str(speaker_dir),
                "utterances": [],
            }

            for audio_path in audio_paths:
                futures.append(
                    executor.submit(
                        load_process_save, audio_path, output_dir_path, speaker_name,
                    )
                )

    for future in tqdm(futures, ncols=0):
        speaker_name, utterance_info = future.result()
        if utterance_info is not None:
            infos["speakers"][speaker_name]["utterances"].append(utterance_info)

    with open(output_dir_path / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
