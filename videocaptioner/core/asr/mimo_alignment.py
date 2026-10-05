"""Strict transcript mapping and cancellable isolated forced alignment for MiMo."""

import io
import json
import math
import os
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

from pydub import AudioSegment

from videocaptioner.config import QWEN3_ALIGNER_MODEL_PATH
from videocaptioner.core.asr.asr_data import ASRDataSeg
from videocaptioner.core.asr.qwen3_runtime import is_model_ready, runtime_python_path
from videocaptioner.core.utils.logger import setup_logger

RUNNER = Path(__file__).with_name("mimo_alignment_runner.py")
logger = setup_logger("mimo_alignment")


def check_mimo_alignment_environment(model_dir=None, runtime_python=None, device="auto"):
    model = Path(model_dir) if model_dir else QWEN3_ALIGNER_MODEL_PATH
    python = Path(runtime_python) if runtime_python else runtime_python_path()
    if not is_model_ready(model):
        return False, "本地 Qwen3-ForcedAligner-0.6B 模型未就绪，请在 MiMo 对齐设置中下载模型。"
    if not python.is_file():
        return False, "本地对齐运行环境未就绪，请在 MiMo 对齐设置中准备运行环境。"
    site = python.parent.parent / "Lib" / "site-packages"
    if os.name != "nt":
        candidates = list((python.parent.parent / "lib").glob("python*/site-packages"))
        site = candidates[0] if candidates else site
    missing = [
        name
        for name in ("torch", "qwen_asr", "soundfile")
        if not (site / name).exists() and not (site / (name + ".py")).exists()
    ]
    if missing:
        return False, "本地对齐运行环境缺少依赖，请在 MiMo 对齐设置中重新准备运行环境。"
    if device not in ("auto", "cpu", "cuda"):
        return False, "本地对齐设备无效，请选择 auto、cpu 或 cuda。"
    if device == "cuda":
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        try:
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
                ],
                capture_output=True,
                timeout=45,
                **kwargs,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, "无法检查本地对齐 CUDA 环境，请检查运行环境或选择 CPU。"
        if probe.returncode:
            return False, "指定了 CUDA 对齐，但运行环境没有可用 CUDA，请选择 CPU 或 auto。"
    return True, "本地强制对齐组件已就绪"


def _clean(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text).casefold()
        if unicodedata.category(c)[0] in "LN" or c == "'"
    )


def _width(text):
    return sum(
        1 if unicodedata.east_asian_width(c) in "WF" else 0.5 for c in text if not c.isspace()
    )


def map_alignment(segment):
    """Restore original punctuation and digits using retained-character positions."""
    text, origin, stop = segment["text"], int(segment["start"]), int(segment["end"])
    if origin < 0 or stop <= origin:
        raise ValueError("MiMo 对齐输入时间范围无效")
    positions = [index for index, char in enumerate(text) for _ in _clean(char)]
    original = _clean(text)
    tokens = segment["tokens"]
    if not original or "".join(_clean(item["text"]) for item in tokens) != original:
        raise ValueError("MiMo 对齐结果未完整映射原文，已停止导出，未使用估算时间。")
    cursor, previous_end, atoms, prefix = 0, 0, [], ""
    pending_start, pending_end = None, None
    for item in tokens:
        cleaned = _clean(item["text"])
        if not cleaned:
            continue
        cursor += len(cleaned)
        text_end = positions[cursor] if cursor < len(positions) else len(text)
        source = text[previous_end:text_end]
        previous_end = text_end
        values = (float(item["start"]), float(item["end"]))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("MiMo 对齐结果包含无效时间")
        start = max(origin, min(stop, origin + round(values[0] * 1000)))
        end = max(origin, min(stop, origin + round(values[1] * 1000)))
        if end < start:
            raise ValueError("MiMo 对齐结果时间倒置")
        if not source:
            pending_start = start if pending_start is None else min(pending_start, start)
            pending_end = end if pending_end is None else max(pending_end, end)
            continue
        if pending_start is not None:
            start, end = min(start, pending_start), max(end, pending_end)
            pending_start, pending_end = None, None
        if end == start:
            if atoms:
                atoms[-1].text += source
            else:
                prefix += source
            continue
        source = prefix + source
        prefix = ""
        if atoms and start < atoms[-1].end_time:
            # Merge overlapping aligner units; never invent per-character times.
            atoms[-1].text += source
            atoms[-1].end_time = max(atoms[-1].end_time, end)
        else:
            atoms.append(ASRDataSeg(source, start, end))
    if not atoms or prefix or "".join(atom.text for atom in atoms) != text:
        raise ValueError("MiMo 对齐结果没有有效字词时间，已停止导出。")
    return atoms


def group_aligned_segment(segment):
    atoms = map_alignment(segment)
    result, current = [], None
    for atom in atoms:
        if _width(atom.text) > 30 or atom.end_time - atom.start_time > 6000:
            raise ValueError("MiMo 对齐单元过长，无法在可信时间边界拆分，请检查识别原文。")
        if current and (
            _width(current.text + atom.text) > 30
            or atom.end_time - current.start_time > 6000
            or atom.start_time - current.end_time >= 700
        ):
            result.append(current)
            current = None
        if current is None:
            current = ASRDataSeg(atom.text, atom.start_time, atom.end_time)
        else:
            current.text += atom.text
            current.end_time = atom.end_time
        duration = current.end_time - current.start_time
        punctuation = (
            current.text.rstrip().rstrip("\"'”’）】》」』").endswith(tuple("。！？!?；;,.，"))
        )
        if (
            _width(current.text) >= 26
            or duration >= 6000
            or (punctuation and (_width(current.text) >= 18 or duration >= 2000))
        ):
            result.append(current)
            current = None
    if current:
        if (
            result
            and _width(current.text) < 5
            and _width(result[-1].text + current.text) <= 30
            and current.end_time - result[-1].start_time <= 6000
            and current.start_time - result[-1].end_time < 700
        ):
            result[-1].text += current.text
            result[-1].end_time = current.end_time
        else:
            result.append(current)
    return result


def align_mimo_segments(
    audio_input,
    segments,
    model_dir=None,
    runtime_python=None,
    device="auto",
    language="auto",
    cancelled=lambda: False,
    on_progress=None,
    on_process=None,
    allow_vad_fallback=False,
):
    if not segments:
        return []
    ready, message = check_mimo_alignment_environment(model_dir, runtime_python, device)
    if not ready:
        raise RuntimeError(message)
    if cancelled():
        raise RuntimeError("MiMo ASR 任务已被用户取消")
    audio = AudioSegment.from_file(
        io.BytesIO(audio_input) if isinstance(audio_input, bytes) else audio_input
    )
    audio = audio.set_frame_rate(16000).set_channels(1)
    for item in segments:
        if item["start"] < 0 or item["end"] > len(audio) + 1 or item["end"] <= item["start"]:
            raise ValueError("MiMo 原始片段时间超出音频范围")
    with tempfile.TemporaryDirectory(prefix="mimo-align-") as directory:
        directory = Path(directory)
        audio.export(str(directory / "audio.wav"), format="wav")
        manifest = {
            "audio": str(directory / "audio.wav"),
            "segments": segments,
            "language": language,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        command = [
            str(runtime_python or runtime_python_path()),
            str(RUNNER),
            "--manifest",
            str(directory / "manifest.json"),
            "--output",
            str(directory / "aligned.json"),
            "--model",
            str(model_dir or QWEN3_ALIGNER_MODEL_PATH),
            "--device",
            device,
        ]
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = None
        with (
            (directory / "progress.log").open("w", encoding="utf-8") as stdout,
            (directory / "error.log").open("w", encoding="utf-8") as stderr,
        ):
            try:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **kwargs)
                if on_process:
                    on_process(process)
                began = time.monotonic()
                last_progress = -1
                last_notice = began
                if on_progress:
                    on_progress(70, "正在加载本地 ForcedAligner（仅对齐原文）...")
                while process.poll() is None:
                    if cancelled():
                        raise RuntimeError("MiMo ASR 任务已被用户取消")
                    if time.monotonic() - began > 1800 + 120 * len(segments):
                        raise RuntimeError("MiMo 本地强制对齐超时，请检查运行环境或使用 GPU。")
                    lines = (directory / "progress.log").read_text(encoding="utf-8").splitlines()
                    events = []
                    for line in lines:
                        try:
                            event = json.loads(line)
                            if (
                                isinstance(event, dict)
                                and isinstance(event.get("progress"), int)
                                and isinstance(event.get("total"), int)
                                and 0 < event["progress"] <= event["total"]
                            ):
                                events.append(event)
                        except (ValueError, TypeError):
                            continue
                    if (
                        events
                        and events[-1]["progress"] != last_progress
                        and time.monotonic() - last_notice >= 1
                    ):
                        last_notice = time.monotonic()
                        last_progress = events[-1]["progress"]
                        if on_progress:
                            total = events[-1]["total"]
                            on_progress(
                                70 + int(25 * last_progress / total),
                                f"本地强制对齐片段 {last_progress}/{total}...",
                            )
                    time.sleep(0.2)
                if cancelled():
                    raise RuntimeError("MiMo ASR 任务已被用户取消")
                if process.returncode:
                    raise RuntimeError(
                        "MiMo 本地强制对齐失败，请检查模型、运行环境及识别原文；未使用均分时间。"
                    )
            finally:
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                if on_process:
                    on_process(None)
        aligned = json.loads((directory / "aligned.json").read_text(encoding="utf-8"))
        return validate_aligned_segments(segments, aligned, allow_vad_fallback=allow_vad_fallback)


def _can_use_vad_sentence_timing(item):
    """Only a complete, finite all-zero alignment of a brief utterance qualifies."""
    cleaned = _clean(item["text"])
    duration = item["end"] - item["start"]
    tokens = item["tokens"]
    if not (0 < len(cleaned) <= 4 and 0 < duration <= 3000 and tokens):
        return False
    if "".join(_clean(token["text"]) for token in tokens) != cleaned:
        return False
    try:
        spans = [(float(token["start"]), float(token["end"])) for token in tokens]
    except (ValueError, TypeError):
        return False
    return all(
        math.isfinite(start) and math.isfinite(end) and 0 <= start == end <= duration / 1000
        for start, end in spans
    )


def validate_aligned_segments(segments, aligned, allow_vad_fallback=False):
    """Validate unchanged original records; keep authorized VAD sentence timing explicit."""
    if len(aligned) != len(segments):
        raise ValueError("MiMo 对齐结果缺少原始片段")
    output, review_count = [], 0
    for index, (original, item) in enumerate(zip(segments, aligned), 1):
        if any(item.get(key) != original.get(key) for key in ("text", "start", "end")):
            raise ValueError("MiMo 对齐结果与原始片段不一致")
        try:
            group = group_aligned_segment(item)
        except ValueError as error:
            if (
                not allow_vad_fallback
                or item["start"] < 0
                or not _can_use_vad_sentence_timing(item)
            ):
                raise ValueError(f"原始片段 {index}：{error}") from None
            cue = ASRDataSeg(item["text"], item["start"], item["end"])
            # These are measured speech-region boundaries, never word timestamps.
            cue.timing_source = "vad_sentence"
            cue.needs_review = True
            cue.source_segment_index = index
            group = [cue]
            review_count += 1
            logger.warning(
                "MiMo 原始片段 %d [%d–%d ms] 字词对齐为零时长，保留 VAD 句级语音起止时间；需复核。",
                index,
                item["start"],
                item["end"],
            )
        if output and group[0].start_time < output[-1].end_time:
            raise ValueError("MiMo 对齐字幕时间重叠")
        output.extend(group)
    if review_count:
        logger.warning(
            "MiMo 共 %d 个短应答使用 VAD 句级时间，需复核；未生成字词时间。", review_count
        )
    return output
