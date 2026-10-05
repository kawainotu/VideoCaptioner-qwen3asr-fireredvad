"""Restore only Qwen's FireRed parameters through the normal config signals."""

from dataclasses import asdict

from videocaptioner.core.qwen3_vad_defaults import FIRERED_VAD_DEFAULTS


def reset_qwen_firered_defaults(config):
    for name, value in asdict(FIRERED_VAD_DEFAULTS).items():
        config.set(getattr(config, "qwen_asr_firered_vad_" + name), value)
