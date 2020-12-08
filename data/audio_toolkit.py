"""Process audio data.
Originally from: https://github.com/resemble-ai/Resemblyzer
"""

import struct
from pathlib import Path
from typing import Union

import numpy as np
import librosa
from webrtcvad import Vad
from scipy.ndimage.morphology import binary_dilation


class AudioToolkit:
    """Audio toolkit for processing audio data."""

    sample_rate = 16000
    preemph = 0.97
    fft_window_ms = 25
    fft_hop_ms = 10
    n_mels = 40
    vad_window_ms = 30
    vad_moving_average_len = 8
    vad_max_silence_len = 6
    norm_dBFS = -30
    int16_max = (2 ** 15) - 1

    @classmethod
    def preprocess_wav(cls, fpath: Union[str, Path]):
        """Load, resample, normalize and trim a waveform."""
        wav, _ = librosa.load(str(fpath), mono=True, sr=cls.sample_rate)
        wav = cls.normalize_volume(wav)
        wav = cls.trim_silences_by_vad(wav)
        return wav

    @classmethod
    def wav_to_logmel(cls, wav: np.ndarray):
        """Derives a mel spectrogram from a preprocessed audio waveform."""
        wav = librosa.effects.preemphasis(wav)
        mels = librosa.feature.melspectrogram(
            wav,
            cls.sample_rate,
            n_fft=int(cls.sample_rate * cls.fft_window_ms / 1000),
            hop_length=int(cls.sample_rate * cls.fft_hop_ms / 1000),
            n_mels=cls.n_mels,
        )
        mels = np.log(mels + 1e-9)
        return mels.astype(np.float32).T

    @classmethod
    def trim_silences_by_vad(cls, wav: np.ndarray):
        """Trim the silences in between the waveform."""
        # Compute the voice detection window size
        vad_window_len = (cls.vad_window_ms * cls.sample_rate) // 1000

        # Append zeros to the end of audio to have a multiple of the window size
        wav = np.concatenate(
            (wav, np.zeros(vad_window_len - len(wav) % vad_window_len))
        )

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack(
            "%dh" % len(wav), *(np.round(wav * cls.int16_max)).astype(np.int16)
        )

        # Perform voice activation detection
        voice_flags = []
        vad = Vad(mode=3)
        for window_start in range(0, len(wav), vad_window_len):
            window_end = window_start + vad_window_len
            voice_flags.append(
                vad.is_speech(
                    pcm_wave[window_start * 2 : window_end * 2],
                    sample_rate=cls.sample_rate,
                )
            )
        voice_flags = np.array(voice_flags)

        audio_mask = _moving_average(voice_flags, cls.vad_moving_average_len)
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(cls.vad_max_silence_len + 1))
        audio_mask = np.repeat(audio_mask, vad_window_len)

        return wav[audio_mask]

    @classmethod
    def normalize_volume(cls, wav: np.ndarray):
        """Normalize waveform volume."""
        rms = np.sqrt(np.mean((wav * cls.int16_max) ** 2))
        dBFS = 20 * np.log10(rms / cls.int16_max)
        return wav * (10 ** ((cls.norm_dBFS - dBFS) / 20))


def _moving_average(array, width):
    """Smooth the voice detection with a moving average."""
    array_padded = np.concatenate(
        (np.zeros((width - 1) // 2), array, np.zeros(width // 2))
    )
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1 :] / width
