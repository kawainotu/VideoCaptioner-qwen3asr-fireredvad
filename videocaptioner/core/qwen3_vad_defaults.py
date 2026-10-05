"""Shared Qwen3-ASR FireRedVAD defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FireRedVADDefaults:
    """FireRedVAD post-processing values; one frame is 10 milliseconds."""

    smooth_window_size: int
    speech_threshold: float
    min_speech_frame: int
    max_speech_frame: int
    min_silence_frame: int
    merge_silence_frame: int
    extend_speech_frame: int
    chunk_max_frame: int


# Official non-streaming FireRedVAD baseline; existing user settings are retained.
FIRERED_VAD_DEFAULTS = FireRedVADDefaults(
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=20,
    max_speech_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000,
)
