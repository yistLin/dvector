"""Wav2Mel for processing audio data."""

import torch
import torch.nn as nn
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram


class Wav2Mel(nn.Module):
    """Transform audio file into mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int = 16000,
        norm_db: float = -3.0,
        sil_threshold: float = 1.0,
        sil_duration: float = 0.1,
        fft_window_ms: float = 25.0,
        fft_hop_ms: float = 10.0,
        f_min: float = 50.0,
        n_mels: int = 40,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.norm_db = norm_db
        self.sil_threshold = sil_threshold
        self.sil_duration = sil_duration
        self.fft_window_ms = fft_window_ms
        self.fft_hop_ms = fft_hop_ms
        self.f_min = f_min
        self.n_mels = n_mels

        self.sox_effects = SoxEffects(sample_rate, norm_db, sil_threshold, sil_duration)
        self.log_melspectrogram = LogMelspectrogram(
            sample_rate, fft_window_ms, fft_hop_ms, f_min, n_mels
        )

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor = self.sox_effects(wav_tensor, sample_rate)
        mel_tensor = self.log_melspectrogram(wav_tensor)
        return mel_tensor


class SoxEffects(nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        sample_rate: int,
        norm_db: float,
        sil_threshold: float,
        sil_duration: float,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
            ["rate", f"{sample_rate}"],  # resample
            ["norm", f"{norm_db}"],  # normalize to -3 dB
            [
                "silence",
                "1",
                f"{sil_duration}",
                f"{sil_threshold}%",
                "-1",
                f"{sil_duration}",
                f"{sil_threshold}%",
            ],  # remove silence throughout the file
        ]

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        return wav_tensor


class LogMelspectrogram(nn.Module):
    """Transform waveform tensors into log mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int,
        fft_window_ms: float,
        fft_hop_ms: float,
        f_min: float,
        n_mels: int,
    ):
        super().__init__()
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=int(sample_rate * fft_hop_ms / 1000),
            n_fft=int(sample_rate * fft_window_ms / 1000),
            f_min=f_min,
            n_mels=n_mels,
        )

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        mel_tensor = self.melspectrogram(wav_tensor).squeeze(0).T  # (time, n_mels)
        return torch.log(torch.clamp(mel_tensor, min=1e-9))
