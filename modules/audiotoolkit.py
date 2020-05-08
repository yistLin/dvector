#!python
# -*- coding: utf-8 -*-
"""Preprocess audio file and data."""

import librosa
import yaml
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class AudioToolkit:
    """Load and process audio data."""

    def __init__(self,
                 sample_rate=16000,
                 min_db_to_trim=10,
                 negative_cutoff_db=80,
                 n_fft=400,
                 n_mels=128,
                 f_min=0,
                 f_max=None,
                 win_len=None,
                 hop_len=None):

        self.sample_rate = sample_rate
        self.min_db_to_trim = min_db_to_trim
        self.negative_cutoff_db = negative_cutoff_db
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.win_len = win_len if win_len is not None else self.n_fft
        self.hop_len = hop_len if hop_len is not None else self.win_len // 2

    @classmethod
    def load_config_file(cls, config_path):
        """Init with given config."""

        with open(config_path) as config_file:
            configs = yaml.load(config_file, Loader=yaml.FullLoader)

        return cls(**configs)

    def file_to_wav_ndarray(self, file_path):
        """Ndarray from the path to audio file."""

        y, _ = librosa.load(file_path, sr=self.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=self.min_db_to_trim)

        return y

    def file_to_wav_tensor(self, file_path):
        """Tensor from the path to audio file."""

        ndarray = self.file_to_wav_ndarray(file_path)

        return torch.from_numpy(ndarray)

    def amplitude_to_db(self, tensor):
        """Convert from power/amplitude scale to decibel scale."""

        return AmplitudeToDB('magnitude',
                             top_db=self.negative_cutoff_db)(tensor)

    def wav_tensor_to_mel(self, tensor, normalization=True, to_db=True):
        """Mel Spectrogram from raw audio tensor.

        Returns:
            - tensor of shape (n_frames, n_mels)
        """

        if normalization:
            tensor = tensor / tensor.norm(float('inf'))

        mel = MelSpectrogram(sample_rate=self.sample_rate,
                             hop_length=self.hop_len,
                             n_fft=self.n_fft,
                             n_mels=self.n_mels)(tensor)

        if to_db:
            mel_db = self.amplitude_to_db(mel)

        if normalization:
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        return mel_db.T

    def file_to_mel_tensor(self, file_path, normalization=True, to_db=True):
        """Tensor of melspectrogram from the path to audio file."""

        tensor = self.file_to_wav_tensor(file_path)

        return self.wav_tensor_to_mel(tensor, normalization, to_db)
