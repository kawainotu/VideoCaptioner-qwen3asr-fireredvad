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


# Tuned against Japanese anime dialogue to retain short utterances and enough
# surrounding context without joining rapid multi-speaker scenes too aggressively.
FIRERED_VAD_DEFAULTS = FireRedVADDefaults(
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=12,
    max_speech_frame=1500,
    min_silence_frame=30,
    merge_silence_frame=50,
    extend_speech_frame=15,
    chunk_max_frame=30000,
)


# Used only to migrate installations that still have every former default.
LEGACY_FIRERED_VAD_DEFAULTS = FireRedVADDefaults(
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=20,
    max_speech_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000,
)
