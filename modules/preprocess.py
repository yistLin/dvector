#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import librosa
import numpy as np

from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel


class Preprocessor:
    """Preprocessor audio data."""

    def __init__(self):
        self.sample_rate = 16000
        self.fft_len = 1024
        self.hop_len = 256
        self.mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        self.min_level = np.exp(-100 / 20 * np.log(10))
        self.top_db = 20

    def butter_highpass(self, wav, cutoff=30, order=5):
        """Apply butter highpass filter"""

        normal_cutoff = cutoff / (0.5 * self.sample_rate)
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        wav = signal.filtfilt(b, a, wav)

        return wav

    def short_time_fourier_transform(self, wav):
        """Apply short-time fourier transform on WAV."""

        wav = np.pad(wav, int(self.fft_len // 2), mode='reflect')

        noverlap = self.fft_len - self.hop_len

        shape = wav.shape[:-1] + ((wav.shape[-1] - noverlap) // self.hop_len,
                                  self.fft_len)
        strides = wav.strides[:-1] + (self.hop_len * wav.strides[-1],
                                      wav.strides[-1])
        result = np.lib.stride_tricks.as_strided(wav,
                                                 shape=shape,
                                                 strides=strides)

        fft_window = get_window('hann', self.fft_len, fftbins=True)

        result = np.fft.rfft(fft_window * result, n=self.fft_len).T

        return np.abs(result)

    def file2spectrogram(self, file_path):
        """Load audio file and create spectrogram from it."""

        wav = librosa.load(file_path, sr=self.sample_rate)[0]
        wav = librosa.effects.trim(wav, top_db=self.top_db)[0]
        wav = self.butter_highpass(wav)
        wav = wav * 0.96

        D = self.short_time_fourier_transform(wav).T
        D_mel = np.dot(D, self.mel_basis)
        D_db = 20 * np.log10(np.maximum(self.min_level, D_mel)) - 16

        S = np.clip((D_db + 100) / 100, 0, 1)

        return S.astype(np.float32)
