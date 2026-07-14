"""Built-in Qwen3-ASR model variants and their local storage locations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from videocaptioner.config import MODEL_PATH, QWEN3_ASR_MODEL_PATH

DownloadSource = Literal["modelscope", "huggingface"]


@dataclass(frozen=True)
class Qwen3ASRModel:
    """Metadata needed to download and load a Qwen3-ASR checkpoint."""

    key: str
    label: str
    description: str
    model_id: str
    source: DownloadSource
    path: Path
    size: str
    ignore_patterns: tuple[str, ...] = ()


DEFAULT_QWEN3_ASR_MODEL_KEY = "qwen3-asr-1.7b"

QWEN3_ASR_MODELS = (
    Qwen3ASRModel(
        key=DEFAULT_QWEN3_ASR_MODEL_KEY,
        label="通用 1.7B",
        description="官方通用语音识别模型",
        model_id="Qwen/Qwen3-ASR-1.7B",
        source="modelscope",
        path=QWEN3_ASR_MODEL_PATH,
        size="4.7 GB",
    ),
    Qwen3ASRModel(
        key="qwen3-asr-1.7b-ja-anime-galgame",
        label="日语动漫 / Galgame 1.7B",
        description="适合日语动画、视觉小说与 Galgame 配音",
        model_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
        source="huggingface",
        path=MODEL_PATH / "Qwen3-ASR-1.7B-JA-Anime-Galgame",
        size="4.1 GB",
        # The repository also contains checkpoint-recovery files that are not
        # needed for inference and would add several GB to the download.
        ignore_patterns=(
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "training_args.bin",
        ),
    ),
)

_MODELS_BY_KEY = {model.key: model for model in QWEN3_ASR_MODELS}


def get_qwen3_asr_model(key: str | None) -> Qwen3ASRModel:
    """Return a model variant, falling back to the official base checkpoint."""
    return _MODELS_BY_KEY.get(key or "", _MODELS_BY_KEY[DEFAULT_QWEN3_ASR_MODEL_KEY])
