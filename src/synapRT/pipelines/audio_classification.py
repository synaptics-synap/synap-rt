import os
from typing import Any, Callable

import librosa
import logging
import numpy as np
from synap import Tensors

from .base import BasePipeline
from ..constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_TOP_N,
    DEFAULT_EN_PROFILING,
    DEFAULT_PROFILE_WINDOW,
    DEFAULT_VALID_INPUT_TYPES,
    # audio
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_CHUNK_AUDIO_DURATION,
    # mel spectrogram
    DEFAULT_MEL_AXIS_ORDER,
    DEFAULT_MEL_INPUT_FORMAT,
    DEFAULT_MEL_LOG,
)

__all__ = [
    "MelSpecClassificationPipeline",
]

logger = logging.getLogger(__name__)


class MelSpecClassificationPipeline(BasePipeline):
    """
    A pipeline for audio classification for models using log-mel spectrogram inputs.

    :param model: Path to SyNAP model file
    :type model: os.PathLike
    :param handler: An optional callback function to handle post-processed inference results
    :type handler: Callable[[dict[str, Any], float | None], Any], optional
    :param window: Window size for STFT in seconds
    :type window: float
    :param hop: Hop size for STFT in seconds
    :type hop: float
    :param n_fft: Number of FFT points
    :type n_fft: int
    :param min_freq: Minimum frequency for mel filterbank
    :type min_freq: float
    :param max_freq: Maximum frequency for mel filterbank
    :type max_freq: float
    :param infer_params: Additional parameters for the inference pipeline
    :type infer_params: Any
    """

    def __init__(
        self,
        model: os.PathLike,
        handler: Callable[[dict[str, Any], float | None], Any] | None = None,
        window: float = 0.025,
        hop: float = 0.010,
        n_fft: int = 1024,
        min_freq: float = 0,
        max_freq: float = 8000,
        **infer_params: Any
    ):
        super().__init__(
            model,
            handler,
            valid_input_types=DEFAULT_VALID_INPUT_TYPES,
            profile=str(infer_params.get("profile", DEFAULT_EN_PROFILING)).lower() == "true",
            profile_window=int(infer_params.get("profile_window", DEFAULT_PROFILE_WINDOW)),
            sample_rate=int(infer_params.get("audio_sample_rate", DEFAULT_AUDIO_SAMPLE_RATE)),
            chunk_duration=float(infer_params.get("chunk_duration", DEFAULT_CHUNK_AUDIO_DURATION))
        )
        self._top_n: int = int(infer_params.get("top_n", DEFAULT_TOP_N))
        self._confidence: float = float(infer_params.get("confidence", DEFAULT_CONFIDENCE))

        # STFT params
        self._audio_sample_rate=int(infer_params.get("audio_sample_rate", DEFAULT_AUDIO_SAMPLE_RATE))
        self._input_format = infer_params.get("input_format", DEFAULT_MEL_INPUT_FORMAT)
        self._log_mel = str(infer_params.get("log_mel", DEFAULT_MEL_LOG)).lower() == "true"
        self._axis_order = infer_params.get("axis_order", DEFAULT_MEL_AXIS_ORDER)
        self._window_len = int(round(float(infer_params.get("window", window)) * self._audio_sample_rate))
        self._hop_len = int(round(float(infer_params.get("hop", hop)) * self._audio_sample_rate))
        self._n_fft = int(infer_params.get("n_fft", n_fft))
        self._min_freq = float(infer_params.get("min_freq", min_freq))
        self._max_freq = float(infer_params.get("max_freq", max_freq))

    def preprocess(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Convert raw int16 PCM audio (mono, 16kHz) into a single log-mel spectrogram
        patch suitable for a YAMNet-style model
        """
        if len(data) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, but got {len(data)}")
        
        raw_audio = data[0]
        if not isinstance(raw_audio, np.ndarray):
            raise TypeError(f"Expected raw audio as np.ndarray, but got {type(raw_audio)}")

        audio_float = raw_audio.astype(np.float32) / 32768.0
        n_frames = self.inputs[0].shape[2]
        n_mels = self.inputs[0].shape[3]

        # STFT to generate power spectrogram
        stft = librosa.stft(
            audio_float,
            n_fft=self._n_fft,
            hop_length=self._hop_len,
            win_length=self._window_len,
            window='hann',
            center=True,
        )
        spectrogram = np.abs(stft)

        # mel filterbank
        mel_spectrogram = librosa.feature.melspectrogram(
            S=spectrogram,
            sr=self._audio_sample_rate,
            n_fft=self._n_fft,
            n_mels=n_mels,
            fmin=self._min_freq,
            fmax=self._max_freq
        )

        # convert to log scale
        if self._log_mel:
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        else:
            log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)

        # swap time/mel axes if needed
        if self._axis_order == "mel_first":
            pass  # i.e. do nothing
        else:
            log_mel_spectrogram = log_mel_spectrogram.T

        # enforce fixed number of frames in the spectrogram dimension
        mel_frames = log_mel_spectrogram.shape[0]
        if mel_frames < n_frames:
            pad_width = n_frames - mel_frames
            pad_value = -80.0 if self._log_mel else 0.0
            log_mel_spectrogram = np.pad(
                log_mel_spectrogram,
                ((0, pad_width), (0, 0)),
                mode='constant',
                constant_values=pad_value
            )
            logger.info(f"Added {pad_width} frames of padding to log-mel spectrogram")
        elif mel_frames > n_frames:
            log_mel_spectrogram = log_mel_spectrogram[:n_frames, :]
            logger.info(f"Truncated {mel_frames - n_frames} frames from log-mel spectrogram")

        # reshape to match the desired model input format
        if self._input_format == "nchw":
            log_mel_spectrogram = log_mel_spectrogram[np.newaxis, np.newaxis, :, :] # [batch=1, channels=1, frames, mels]
        elif self._input_format == "bft":
            log_mel_spectrogram = log_mel_spectrogram[np.newaxis, :, :]             # [batch=1, frames, mels]
        elif self._input_format == "nhwc":
            log_mel_spectrogram = log_mel_spectrogram[np.newaxis, :, :, np.newaxis] # [batch=1, frames, mels, channels=1]
        res = log_mel_spectrogram.astype(self.inputs[0].data_type.np_type())
    
        return [res]


    def postprocess(self, data: Tensors) -> None:
        """
        Post-process the output of the model to get the top N predictions.
        """
        # get top N predictions
        probs = data[0].to_numpy().flatten()
        predicted_indices = np.where(probs >= self._confidence)[0]
        predicted_indices = predicted_indices[np.argsort(probs[predicted_indices])[::-1]]
        self._results = {int(idx): float(probs[idx]) for idx in predicted_indices[:self._top_n]}
