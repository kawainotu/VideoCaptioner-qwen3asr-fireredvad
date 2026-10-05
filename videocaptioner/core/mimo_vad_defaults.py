"""MiMo FireRed defaults: official non-streaming baseline, 2026-10-04.

Source: https://github.com/FireRedTeam/FireRedVAD/blob/main/README.md
These defaults are independent of Qwen's tuned configuration.
"""

MIMO_FIRERED_VAD_DEFAULTS = {
    "smooth_window_size": 5,
    "speech_threshold": 0.4,
    "min_speech_frame": 20,
    "max_speech_frame": 2000,
    "min_silence_frame": 20,
    "merge_silence_frame": 0,
    "extend_speech_frame": 0,
    "chunk_max_frame": 30000,
}
