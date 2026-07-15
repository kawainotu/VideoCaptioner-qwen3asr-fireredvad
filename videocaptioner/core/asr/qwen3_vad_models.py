"""VAD model choices available to the isolated Qwen3-ASR runner."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from videocaptioner.config import (
    BUNDLED_FIRERED_VAD_MODEL_PATH,
    FIRERED_VAD_MODEL_PATH,
)

DownloadSource = Literal["huggingface"]


@dataclass(frozen=True)
class Qwen3VADModel:
    key: str
    label: str
    description: str
    model_id: Optional[str]
    source: Optional[DownloadSource]
    path: Optional[Path]
    size: str
    ignore_patterns: tuple[str, ...] = ()


DEFAULT_QWEN3_VAD_MODEL_KEY = "silero"
FIRERED_VAD_MODEL_KEY = "firered"

QWEN3_VAD_MODELS = (
    Qwen3VADModel(
        key=DEFAULT_QWEN3_VAD_MODEL_KEY,
        label="Silero VAD",
        description="轻量、稳定，适合通用语音活动检测",
        model_id=None,
        source=None,
        path=None,
        size="随运行环境安装",
    ),
    Qwen3VADModel(
        key=FIRERED_VAD_MODEL_KEY,
        label="FireRedVAD",
        description="多语言离线 VAD，支持更完整的帧级切分控制",
        model_id="FireRedTeam/FireRedVAD",
        source="huggingface",
        path=FIRERED_VAD_MODEL_PATH,
        size="2.4 MB",
        ignore_patterns=("AED/*", "Stream-VAD/*"),
    ),
)

_MODELS_BY_KEY = {model.key: model for model in QWEN3_VAD_MODELS}


def get_qwen3_vad_model(key: str | None) -> Qwen3VADModel:
    """Return a VAD choice, falling back to Silero for old configurations."""
    return _MODELS_BY_KEY.get(key or "", _MODELS_BY_KEY[DEFAULT_QWEN3_VAD_MODEL_KEY])


def _firered_weights_dir(path: Path) -> Path:
    if (path / "model.pth.tar").is_file() or (path / "cmvn.ark").is_file():
        return path
    return path / "VAD"


def is_firered_vad_model_ready(path: Optional[Path] = None) -> bool:
    """Check either an explicit FireRed VAD folder or all supported locations."""
    candidates = (
        (path,)
        if path is not None
        else (FIRERED_VAD_MODEL_PATH, BUNDLED_FIRERED_VAD_MODEL_PATH)
    )
    for candidate in candidates:
        weights_dir = _firered_weights_dir(candidate)
        cmvn = weights_dir / "cmvn.ark"
        weights = weights_dir / "model.pth.tar"
        if (
            cmvn.is_file()
            and cmvn.stat().st_size > 0
            and weights.is_file()
            and weights.stat().st_size > 0
        ):
            return True
    return False


def resolve_firered_vad_model_path() -> Path:
    """Prefer an updated user model, then the copy embedded in packaged builds."""
    if is_firered_vad_model_ready(FIRERED_VAD_MODEL_PATH):
        return _firered_weights_dir(FIRERED_VAD_MODEL_PATH)
    if is_firered_vad_model_ready(BUNDLED_FIRERED_VAD_MODEL_PATH):
        return _firered_weights_dir(BUNDLED_FIRERED_VAD_MODEL_PATH)
    return _firered_weights_dir(FIRERED_VAD_MODEL_PATH)
