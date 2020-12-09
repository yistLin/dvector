"""AudioToolkit for processing audio data."""

from pathlib import Path
from typing import Union

import numpy as np
import librosa
from sox import Transformer


class AudioToolkit:
    """Audio toolkit for processing audio data."""

    sample_rate = 16000
    preemph = 0.97
    fft_window_ms = 25
    fft_hop_ms = 10
    n_mels = 40

    @classmethod
    def preprocess_wav(cls, fpath: Union[str, Path]) -> np.ndarray:
        """Load, resample, normalize and trim a waveform."""
        transformer = Transformer()
        transformer.norm()
        transformer.silence(silence_threshold=1, min_silence_duration=0.1)
        transformer.set_output_format(rate=cls.sample_rate, bits=16, channels=1)
        wav = transformer.build_array(input_filepath=str(fpath))
        wav = wav / (2 ** 15)
        return wav.astype(np.float32)

    @classmethod
    def wav_to_logmel(cls, wav: np.ndarray) -> np.ndarray:
        """Derives a log mel spectrogram from a preprocessed waveform."""
        wav = librosa.effects.preemphasis(wav)
        mel = librosa.feature.melspectrogram(
            wav,
            cls.sample_rate,
            n_fft=int(cls.sample_rate * cls.fft_window_ms / 1000),
            hop_length=int(cls.sample_rate * cls.fft_hop_ms / 1000),
            n_mels=cls.n_mels,
        )
        mel = np.log(mel + 1e-9)
        return mel.astype(np.float32).T
