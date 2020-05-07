#!python
# -*- coding: utf-8 -*-
"""Preprocess audio file and data."""

import librosa
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class AudioToolkit:
    """Load and process audio data."""

    sample_rate = 16000
    min_db_to_trim = 10
    negative_cutoff_db = 80
    n_fft = 512
    n_mels = 40
    hop_len = 256

    @classmethod
    def file_to_ndarray(cls, file_path):
        """Ndarray from the path to audio file."""

        y, _ = librosa.load(file_path, sr=cls.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=cls.min_db_to_trim)

        return y

    @classmethod
    def file_to_tensor(cls, file_path):
        """Tensor from the path to audio file."""

        ndarray = cls.file_to_ndarray(file_path)

        return torch.from_numpy(ndarray)

    @classmethod
    def normalize_wav(cls, tensor):
        """Normalize a raw audio tensor"""

        return tensor / tensor.norm(float('inf'))

    @classmethod
    def amplitude_to_db(cls, tensor):
        """Convert from power/amplitude scale to decibel scale."""

        return AmplitudeToDB('magnitude',
                             top_db=cls.negative_cutoff_db)(tensor)

    @classmethod
    def wav_to_mel(cls, tensor, normalization=True, to_db=True):
        """Mel Spectrogram from raw audio tensor.

        Returns:
            - tensor of shape (n_frames, n_mels)
        """

        if normalization:
            tensor = cls.normalize_wav(tensor)

        mel = MelSpectrogram(sample_rate=cls.sample_rate,
                             hop_length=cls.hop_len,
                             n_fft=cls.n_fft,
                             n_mels=cls.n_mels)(tensor)

        if to_db:
            mel = cls.amplitude_to_db(mel)

        return mel.T

    @classmethod
    def file_to_mel(cls, file_path, normalization=True, to_db=True):
        """Tensor of melspectrogram from the path to audio file."""

        tensor = cls.file_to_tensor(file_path)

        return cls.wav_to_mel(tensor, normalization, to_db)
