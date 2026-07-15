import json
import os
from pathlib import Path
from typing import Optional

from videocaptioner.config import (
    QWEN3_ALIGNER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
    QWEN3_ASR_RUNTIME_PATH,
)
from videocaptioner.core.asr.qwen3_vad_models import (
    FIRERED_VAD_MODEL_KEY,
    get_qwen3_vad_model,
    is_firered_vad_model_ready,
)

RUNTIME_VERSION = 2
QWEN_ASR_PACKAGE = "qwen-asr==0.0.6"
SILERO_VAD_PACKAGE = "silero-vad>=6.0,<7"
FIRERED_VAD_PACKAGE = "fireredvad==0.0.2"


def runtime_python_path(runtime_dir: Optional[Path] = None) -> Path:
    root = runtime_dir or QWEN3_ASR_RUNTIME_PATH
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def runtime_marker_path(runtime_dir: Optional[Path] = None) -> Path:
    return (runtime_dir or QWEN3_ASR_RUNTIME_PATH) / "videocaptioner-runtime.json"


def is_runtime_ready(runtime_dir: Optional[Path] = None) -> bool:
    python_path = runtime_python_path(runtime_dir)
    marker_path = runtime_marker_path(runtime_dir)
    if not python_path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return marker.get("runtime_version") == RUNTIME_VERSION


def write_runtime_marker(runtime_dir: Optional[Path] = None) -> None:
    marker_path = runtime_marker_path(runtime_dir)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "runtime_version": RUNTIME_VERSION,
                "qwen_asr": QWEN_ASR_PACKAGE,
                "silero_vad": SILERO_VAD_PACKAGE,
                "firered_vad": FIRERED_VAD_PACKAGE,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def is_model_ready(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    return any(path.stat().st_size > 0 for path in model_dir.glob("*.safetensors"))


def missing_components(
    runtime_dir: Optional[Path] = None,
    asr_model_dir: Optional[Path] = None,
    aligner_model_dir: Optional[Path] = None,
    vad_model_key: Optional[str] = None,
    vad_model_dir: Optional[Path] = None,
) -> list[str]:
    missing = []
    if not is_runtime_ready(runtime_dir):
        missing.append("Qwen3-ASR runtime")
    if not is_model_ready(asr_model_dir or QWEN3_ASR_MODEL_PATH):
        missing.append("Qwen3-ASR-1.7B")
    if not is_model_ready(aligner_model_dir or QWEN3_ALIGNER_MODEL_PATH):
        missing.append("Qwen3-ForcedAligner-0.6B")
    vad_model = get_qwen3_vad_model(vad_model_key)
    if (
        vad_model_key is not None
        and vad_model.key == FIRERED_VAD_MODEL_KEY
        and not is_firered_vad_model_ready(vad_model_dir)
    ):
        missing.append("FireRedVAD")
    return missing
