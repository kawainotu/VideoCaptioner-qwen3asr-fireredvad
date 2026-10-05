"""Speech detection (VAD) infrastructure for MiMo ASR.

Reuses the existing local VAD runtime infrastructure without loading Qwen ASR
or requiring any recognition models.
"""

import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from videocaptioner.core.asr.mimo_vad_runner import FIRERED_DEFAULTS
from videocaptioner.core.asr.qwen3_runtime import runtime_python_path
from videocaptioner.core.asr.qwen3_vad_models import (
    is_firered_vad_model_ready,
    resolve_firered_vad_model_path,
)
from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("mimo_vad")

RUNNER_SCRIPT_PATH = Path(__file__).with_name("mimo_vad_runner.py")


def check_vad_environment(
    custom_runtime_python: Optional[Path] = None,
    vad_model: str = "silero",
    model_dir: Optional[str] = None,
    check_model: bool = True,
) -> tuple[bool, str]:
    """Check selected VAD dependencies and weights without importing recognition models."""
    if vad_model not in ("silero", "firered"):
        return False, f"Unsupported VAD model: {vad_model}"
    if (
        check_model
        and vad_model == "firered"
        and not is_firered_vad_model_ready(Path(model_dir) if model_dir else None)
    ):
        return False, "FireRedVAD 模型未安装，请在 MiMo VAD 设置中下载模型，或关闭 VAD 过滤。"
    required = (
        ("torch", "soundfile", "numpy", "librosa", "fireredvad", "kaldi_native_fbank")
        if vad_model == "firered"
        else ("silero_vad", "torch", "soundfile", "librosa")
    )
    if custom_runtime_python is None:
        try:
            if all(importlib.util.find_spec(pkg) is not None for pkg in required):
                return True, f"当前 Python 环境的 {vad_model} VAD 已就绪"
        except (ImportError, ValueError):
            pass
    py_path = custom_runtime_python or runtime_python_path()
    if py_path.is_file():
        site_packages = py_path.parent.parent / "Lib" / "site-packages"
        if not site_packages.is_dir():
            return False, f"未检测到独立环境的 site-packages 目录: {site_packages}"
        missing = [
            pkg
            for pkg in required
            if not (site_packages / pkg).exists()
            and not (site_packages / f"{pkg}.py").exists()
            and not any(site_packages.glob(f"{pkg}*"))
        ]
        if not missing:
            return True, f"独立运行环境的 {vad_model} VAD 已就绪"
        return (
            False,
            f"独立运行环境缺少依赖: {', '.join(missing)}。请在 MiMo VAD 设置中重新准备运行环境。",
        )
    return False, "未检测到本地 VAD 运行环境，请在 MiMo VAD 设置中准备环境，或关闭 VAD 过滤。"


def detect_speech_regions(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 300,
    runtime_python: Optional[str] = None,
    vad_model: str = "silero",
    model_dir: Optional[str] = None,
    **firered_options,
) -> list[tuple[float, float]]:
    """Detect speech regions using the selected local VAD model.

    Returns:
        List of (start_seconds, end_seconds) for each detected speech segment.

    Raises:
        RuntimeError: If VAD prerequisites are missing or detection fails.
    """
    audio_file = Path(audio_path)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if vad_model not in ("silero", "firered"):
        raise ValueError(f"Unsupported VAD model: {vad_model}")
    if vad_model == "firered":
        if not is_firered_vad_model_ready(Path(model_dir) if model_dir else None):
            raise RuntimeError("FireRedVAD 模型未安装，请在 MiMo VAD 设置中下载模型。")
        model_path = Path(model_dir) if model_dir else resolve_firered_vad_model_path()
        if not (model_path / "model.pth.tar").is_file():
            model_path = model_path / "VAD"
        model_dir = str(model_path)
    options = dict(
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        **firered_options,
    )
    if runtime_python is None:
        try:
            from videocaptioner.core.asr.mimo_vad_runner import detect_regions

            regions = detect_regions(
                str(audio_file), vad_model=vad_model, model_dir=model_dir, **options
            )
            return [(r["start"], r["end"]) for r in regions]
        except ImportError:
            pass

    # Fall back to isolated runtime python
    py_path = Path(runtime_python) if runtime_python else runtime_python_path()
    if not py_path.is_file():
        is_ready, msg = check_vad_environment(py_path, vad_model=vad_model, model_dir=model_dir)
        raise RuntimeError(msg)

    if not RUNNER_SCRIPT_PATH.is_file():
        raise RuntimeError(f"VAD runner script not found: {RUNNER_SCRIPT_PATH}")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_json = Path(temp_dir) / "speech_regions.json"
        cmd = [
            str(py_path),
            str(RUNNER_SCRIPT_PATH),
            "--audio",
            str(audio_file),
            "--output",
            str(output_json),
            "--threshold",
            str(threshold),
            "--min-speech-ms",
            str(min_speech_ms),
            "--min-silence-ms",
            str(min_silence_ms),
            "--speech-pad-ms",
            str(speech_pad_ms),
            "--vad-model",
            vad_model,
        ]
        if model_dir:
            cmd.extend(["--model-dir", model_dir])
        for key, default in FIRERED_DEFAULTS.items():
            cmd.extend(
                [
                    "--firered-vad-" + key.replace("_", "-"),
                    str(firered_options.get("firered_vad_" + key, default)),
                ]
            )

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        extra_kwargs = {}
        if os.name == "nt":
            extra_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                **extra_kwargs,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "本地 VAD 语音切片检测超时 (超过 600 秒)，可能是长音频切片耗时过长或子进程异常卡死"
            )

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            logger.error("VAD runner failed with code %d: %s", proc.returncode, err)
            raise RuntimeError(f"本地 VAD 检测失败: {err}")

        if not output_json.is_file():
            raise RuntimeError("VAD runner did not produce output file")

        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [(float(item["start"]), float(item["end"])) for item in data]


def merge_speech_regions_into_chunks(
    speech_regions: list[tuple[float, float]],
    max_chunk_duration: float = 25.0,
    max_gap_seconds: Optional[float] = 0.8,
) -> list[tuple[float, float]]:
    """Merge and bound speech regions into chunks without shifting the original timeline.

    Args:
        speech_regions: List of (start_sec, end_sec) detected by VAD.
        max_chunk_duration: Maximum allowed duration for a single chunk (e.g. 20-30s).
        max_gap_seconds: Merge adjacent speech regions if silence gap is <= this value.

    Returns:
        List of bounded (chunk_start_sec, chunk_end_sec) chunks referencing the original timeline.
    """
    if not speech_regions:
        return []

    # First, split any single region that exceeds max_chunk_duration
    bounded_regions: list[tuple[float, float]] = []
    for start, end in speech_regions:
        if end <= start:
            continue
        cur = start
        while cur < end:
            chunk_end = min(cur + max_chunk_duration, end)
            bounded_regions.append((round(cur, 3), round(chunk_end, 3)))
            cur = chunk_end

    if not bounded_regions:
        return []

    # Second, merge adjacent regions if gap is small and total duration <= max_chunk_duration
    chunks: list[tuple[float, float]] = []
    current_start, current_end = bounded_regions[0]

    for next_start, next_end in bounded_regions[1:]:
        gap = next_start - current_end
        combined_duration = next_end - current_start
        if (
            max_gap_seconds is not None
            and gap <= max_gap_seconds
            and combined_duration <= max_chunk_duration
        ):
            current_end = max(current_end, next_end)
        else:
            chunks.append((round(current_start, 3), round(current_end, 3)))
            current_start, current_end = next_start, next_end

    chunks.append((round(current_start, 3), round(current_end, 3)))
    return chunks
