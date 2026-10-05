"""Replay cached MiMo transcripts with the actual isolated aligner, without API calls.

Run with the application's Python. Raw tokens and failures stay in --output-dir;
console output contains only indices, timing statistics and validation errors.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def check_segment(index, item, output, summary, group):
    summary["checked"] += 1
    tokens = item["tokens"]
    stats = {
        "index": index,
        "start_ms": item["start"],
        "end_ms": item["end"],
        "duration_ms": item["end"] - item["start"],
        "characters": len(item["text"]),
        "tokens": len(tokens),
        "positive_tokens": sum(t["end"] > t["start"] for t in tokens),
        "min_token_start": min((t["start"] for t in tokens), default=None),
        "max_token_end": max((t["end"] for t in tokens), default=None),
    }
    try:
        cues = group(item)
        stats.update(cues=len(cues), valid=True)
    except ValueError as error:
        stats.update(valid=False, error=str(error))
        summary["failed"].append(stats)
        write_json(output / f"failure-{index:04d}.json", dict(stats, segment=item))
    with (output / "checks.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(stats, ensure_ascii=False) + "\n")
    write_json(output / "summary.json", summary)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats["valid"]


def worker(args):
    # Load only the standalone runner; the model runtime need not contain GUI deps.
    spec = importlib.util.spec_from_file_location(
        "mimo_diagnostic_runner", ROOT / "videocaptioner/core/asr/mimo_alignment_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = read_json(args.manifest)
    indices = manifest.pop("source_indices")
    originals = manifest["segments"]
    if args.context_ms:
        import soundfile as sf

        info = sf.info(manifest["audio"])
        duration_ms = round(info.frames * 1000 / info.samplerate)
        manifest["segments"] = [
            dict(
                item,
                start=max(0, item["start"] - args.context_ms),
                end=min(duration_ms, item["end"] + args.context_ms),
            )
            for item in originals
        ]

    def save_segment(index, item):
        source_index = indices[index]
        if args.context_ms:
            original = originals[index]
            offset = (original["start"] - item["start"]) / 1000
            item = dict(
                item,
                start=original["start"],
                end=original["end"],
                tokens=[
                    dict(token, start=token["start"] - offset, end=token["end"] - offset)
                    for token in item["tokens"]
                ],
            )
        write_json(Path(args.output_dir) / f"segment-{source_index:04d}.json", item)
        print(json.dumps({"saved_segment": source_index}), flush=True)

    module.align_manifest(
        manifest,
        args.model,
        args.device,
        on_segment=save_segment,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-python")
    parser.add_argument("--model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--short-first", action="store_true")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--indices", help="Comma-separated one-based original segment indices")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--aligned", help="Validate an existing aligned JSON without loading a model"
    )
    parser.add_argument(
        "--context-ms", type=int, default=0, help="Experimental audio context padding"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    sys.path.insert(0, str(ROOT))
    from videocaptioner.config import QWEN3_ALIGNER_MODEL_PATH
    from videocaptioner.core.asr.mimo_alignment import group_aligned_segment
    from videocaptioner.core.asr.qwen3_runtime import runtime_python_path

    manifest = read_json(args.manifest)
    selected = list(enumerate(manifest["segments"], 1))
    if args.indices:
        indices = {int(index) for index in args.indices.split(",")}
        if not indices or min(indices) < 1 or max(indices) > len(selected):
            parser.error("--indices must refer to existing one-based segment indices")
        selected = [(index, item) for index, item in selected if index in indices]
    if args.short_first:
        selected.sort(key=lambda entry: (entry[1]["end"] - entry[1]["start"], entry[0]))
    if args.max_segments is not None:
        if args.max_segments < 1:
            parser.error("--max-segments must be positive")
        selected = selected[: args.max_segments]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"selected": len(selected), "checked": 0, "failed": []}
    if args.aligned:
        aligned = read_json(args.aligned)
        if len(aligned) != len(manifest["segments"]):
            parser.error("Aligned result has a different segment count")
        for index, original in selected:
            item = aligned[index - 1]
            if any(item.get(key) != original.get(key) for key in ("text", "start", "end")):
                parser.error(f"Aligned result differs from original segment {index}")
            valid = check_segment(index, item, output, summary, group_aligned_segment)
            if not valid and not args.continue_on_error:
                break
        if summary["failed"]:
            raise SystemExit(1)
        return
    replay = dict(
        manifest,
        segments=[item for _, item in selected],
        source_indices=[index for index, _ in selected],
    )
    write_json(output / "replay-manifest.json", replay)
    command = [
        str(args.runtime_python or runtime_python_path()),
        str(Path(__file__).resolve()),
        "--worker",
        "--manifest",
        str(output / "replay-manifest.json"),
        "--output-dir",
        str(output),
        "--model",
        str(args.model or QWEN3_ALIGNER_MODEL_PATH),
        "--device",
        args.device,
        "--context-ms",
        str(args.context_ms),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    print(
        f"Replaying {len(selected)} cached segments on {args.device}; no API requests", flush=True
    )
    with (output / "runner-error.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            **kwargs,
        )
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict) or "saved_segment" not in event:
                    continue
                index = event["saved_segment"]
                item = read_json(output / f"segment-{index:04d}.json")
                valid = check_segment(index, item, output, summary, group_aligned_segment)
                if not valid and not args.continue_on_error:
                    break
            else:
                process.wait()
                summary["runner_returncode"] = process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            write_json(output / "summary.json", summary)
    if summary["failed"] or summary.get("runner_returncode", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
