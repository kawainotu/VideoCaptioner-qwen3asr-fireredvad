"""Isolated Qwen3-ASR worker.

This module intentionally avoids imports from VideoCaptioner so it can run in
the dedicated Qwen virtual environment.
"""

# pyright: reportMissingImports=false

import argparse
import gc
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 170
MERGE_GAP_SECONDS = 2.0
ALIGNER_LANGUAGES = {
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
}


@dataclass
class AudioChunk:
    audio: Any
    offset_seconds: float


@dataclass
class TranscribedChunk:
    audio: Any
    offset_seconds: float
    text: str
    language: str


def emit(event_type: str, **payload: Any) -> None:
    print(
        json.dumps({"type": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def progress(value: int, message: str) -> None:
    emit("progress", progress=max(0, min(100, value)), message=message)


def resolve_device(torch_module: Any, requested: str) -> tuple[str, Any]:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")
    use_cuda = requested == "cuda" or (
        requested == "auto" and torch_module.cuda.is_available()
    )
    if use_cuda and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was selected but is not available in the Qwen runtime")
    if not use_cuda:
        return "cpu", torch_module.float32
    if torch_module.cuda.is_bf16_supported():
        return "cuda:0", torch_module.bfloat16
    return "cuda:0", torch_module.float16


def release_memory(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.ipc_collect()


def fixed_chunks(audio: Any) -> list[AudioChunk]:
    max_samples = MAX_CHUNK_SECONDS * SAMPLE_RATE
    chunks = []
    for start in range(0, len(audio), max_samples):
        end = min(start + max_samples, len(audio))
        if end - start < SAMPLE_RATE // 2:
            if chunks:
                previous = chunks[-1]
                previous.audio = __import__("numpy").concatenate(
                    [previous.audio, audio[start:end]]
                )
            continue
        chunks.append(AudioChunk(audio[start:end], start / SAMPLE_RATE))
    return chunks


def merge_speech_regions(regions: list[dict[str, int]], audio: Any) -> list[AudioChunk]:
    if not regions:
        return []
    max_samples = MAX_CHUNK_SECONDS * SAMPLE_RATE
    max_gap = int(MERGE_GAP_SECONDS * SAMPLE_RATE)
    merged: list[tuple[int, int]] = []
    start = int(regions[0]["start"])
    end = int(regions[0]["end"])
    for region in regions[1:]:
        next_start = int(region["start"])
        next_end = int(region["end"])
        if next_start - end <= max_gap and next_end - start <= max_samples:
            end = next_end
            continue
        merged.append((start, end))
        start, end = next_start, next_end
    merged.append((start, end))

    chunks: list[AudioChunk] = []
    for start, end in merged:
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_samples, end)
            if chunk_end - cursor >= SAMPLE_RATE // 2:
                chunks.append(AudioChunk(audio[cursor:chunk_end], cursor / SAMPLE_RATE))
            cursor = chunk_end
    return chunks


def detect_speech_chunks(audio: Any, args: argparse.Namespace, torch_module: Any) -> list[AudioChunk]:
    if not args.vad_filter:
        return fixed_chunks(audio)

    from silero_vad import get_speech_timestamps, load_silero_vad

    progress(5, "Loading Silero VAD")
    vad_model = load_silero_vad(onnx=False)
    waveform = torch_module.from_numpy(audio).float()
    regions = get_speech_timestamps(
        waveform,
        vad_model,
        threshold=args.vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        return_seconds=False,
    )
    del vad_model, waveform
    gc.collect()
    progress(10, f"VAD found {len(regions)} speech regions")
    return merge_speech_regions(regions, audio)


def load_asr_model(args: argparse.Namespace, device: str, dtype: Any) -> Any:
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(
        args.asr_model,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=1024,
    )


def load_aligner_model(args: argparse.Namespace, device: str, dtype: Any) -> Any:
    from qwen_asr import Qwen3ForcedAligner

    return Qwen3ForcedAligner.from_pretrained(
        args.aligner_model,
        dtype=dtype,
        device_map=device,
    )


def transcribe_chunks(
    chunks: list[AudioChunk],
    model: Any,
    args: argparse.Namespace,
) -> list[TranscribedChunk]:
    results: list[TranscribedChunk] = []
    total = max(len(chunks), 1)
    # qwen_asr accepts canonical language names, but the rest of the
    # application stores ISO 639 codes (for example, "ja" and "pt").
    language = normalize_language(args.language)
    for index, chunk in enumerate(chunks):
        item = model.transcribe(
            audio=(chunk.audio, SAMPLE_RATE),
            context=args.context or "",
            language=language or None,
            return_time_stamps=False,
        )[0]
        results.append(
            TranscribedChunk(
                audio=chunk.audio,
                offset_seconds=chunk.offset_seconds,
                text=item.text.strip(),
                language=item.language,
            )
        )
        progress(20 + int(38 * (index + 1) / total), f"Transcribed {index + 1}/{total}")
    return results


def normalize_language(language: str) -> str:
    if not language:
        return ""
    first = language.split(",", 1)[0].strip()
    aliases = {
        "zh": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "ar": "Arabic",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "id": "Indonesian",
        "it": "Italian",
        "ko": "Korean",
        "ru": "Russian",
        "th": "Thai",
        "vi": "Vietnamese",
        "ja": "Japanese",
        "tr": "Turkish",
        "hi": "Hindi",
        "ms": "Malay",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "pl": "Polish",
        "cs": "Czech",
        "tl": "Filipino",
        "fa": "Persian",
        "el": "Greek",
        "ro": "Romanian",
        "hu": "Hungarian",
        "mk": "Macedonian",
    }
    return aliases.get(first.lower(), first)


def approximate_units(text: str) -> list[str]:
    pattern = re.compile(
        r"[A-Za-zÀ-ž'’-]+|\d+(?:[.,]\d+)*|[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|[^\s]"
    )
    return pattern.findall(text)


def approximate_segments(chunk: TranscribedChunk) -> list[dict[str, Any]]:
    units = approximate_units(chunk.text)
    if not units:
        return []
    duration_ms = max(1, int(len(chunk.audio) * 1000 / SAMPLE_RATE))
    offset_ms = int(round(chunk.offset_seconds * 1000))
    result = []
    for index, unit in enumerate(units):
        start = offset_ms + int(duration_ms * index / len(units))
        end = offset_ms + int(duration_ms * (index + 1) / len(units))
        result.append({"text": unit, "start_time": start, "end_time": max(start + 1, end)})
    return result


def align_chunks(
    chunks: list[TranscribedChunk],
    aligner: Any,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    total = max(len(chunks), 1)
    for index, chunk in enumerate(chunks):
        if not chunk.text:
            continue
        language = normalize_language(chunk.language)
        if language not in ALIGNER_LANGUAGES:
            segments.extend(approximate_segments(chunk))
            progress(
                65 + int(30 * (index + 1) / total),
                f"Used approximate timestamps for {language or 'unknown language'}",
            )
            continue
        result = aligner.align(
            audio=(chunk.audio, SAMPLE_RATE),
            text=chunk.text,
            language=language,
        )[0]
        offset_ms = int(round(chunk.offset_seconds * 1000))
        for item in result:
            start = offset_ms + int(round(float(item.start_time) * 1000))
            end = offset_ms + int(round(float(item.end_time) * 1000))
            segments.append(
                {
                    "text": item.text,
                    "start_time": start,
                    "end_time": max(start + 1, end),
                }
            )
        progress(65 + int(30 * (index + 1) / total), f"Aligned {index + 1}/{total}")
    return segments


def run(args: argparse.Namespace) -> dict[str, Any]:
    import librosa
    import numpy as np
    import torch

    progress(1, "Loading audio")
    audio, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    chunks = detect_speech_chunks(audio, args, torch)
    if not chunks:
        progress(100, "No speech detected")
        return {"segments": [], "language": ""}

    device, dtype = resolve_device(torch, args.device)
    progress(12, f"Loading Qwen3-ASR on {device}")
    asr_model = load_asr_model(args, device, dtype)
    aligner = None
    if not args.low_memory:
        progress(16, f"Loading forced aligner on {device}")
        aligner = load_aligner_model(args, device, dtype)

    transcribed = transcribe_chunks(chunks, asr_model, args)
    if args.low_memory:
        progress(60, "Releasing ASR model from memory")
        del asr_model
        asr_model = None
        release_memory(torch)
        progress(62, f"Loading forced aligner on {device}")
        aligner = load_aligner_model(args, device, dtype)

    assert aligner is not None
    segments = align_chunks(transcribed, aligner)
    segments.sort(key=lambda item: (item["start_time"], item["end_time"]))
    aligner = None
    asr_model = None
    release_memory(torch)
    progress(98, "Writing timestamped transcript")
    languages = sorted({chunk.language for chunk in transcribed if chunk.language})
    return {"segments": segments, "language": ",".join(languages)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--aligner-model", required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--language", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=500)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = run(args)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        progress(100, "Qwen3-ASR completed")
        emit("result", path=args.output)
        return 0
    except Exception as exc:
        emit("error", message=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
