"""Validate and export cached MiMo transcripts through the production alignment entry point."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--runtime-python")
    parser.add_argument("--model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-vad-fallback", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from videocaptioner.core.asr.asr_data import ASRData
    from videocaptioner.core.asr.mimo_alignment import _width, align_mimo_segments

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cues = align_mimo_segments(
        manifest["audio"],
        manifest["segments"],
        model_dir=args.model,
        runtime_python=args.runtime_python,
        device=args.device,
        language=manifest.get("language", "auto"),
        on_progress=lambda value, message: print(f"{value}% {message}", flush=True),
        allow_vad_fallback=args.allow_vad_fallback,
    )
    assert "".join(cue.text for cue in cues) == "".join(
        item["text"] for item in manifest["segments"]
    ), "Original text changed"
    assert all(0 <= cue.start_time < cue.end_time for cue in cues), "Invalid cue timing"
    assert all(_width(cue.text) <= 30 for cue in cues), "Cue exceeds width limit"
    assert all(cue.end_time - cue.start_time <= 6000 for cue in cues), "Cue exceeds duration limit"
    assert all(a.end_time <= b.start_time for a, b in zip(cues, cues[1:])), "Overlapping cues"
    output = Path(args.srt)
    output.parent.mkdir(parents=True, exist_ok=True)
    ASRData(cues).save(str(output))
    summary = {
        "original_segments": len(manifest["segments"]),
        "subtitle_cues": len(cues),
        "original_text_preserved": True,
        "nonoverlapping": True,
        "maximum_width": max((_width(cue.text) for cue in cues), default=0),
        "maximum_duration_ms": max((cue.end_time - cue.start_time for cue in cues), default=0),
        "first_start_ms": cues[0].start_time if cues else None,
        "last_end_ms": cues[-1].end_time if cues else None,
        "srt": str(output.resolve()),
        "needs_review": [
            {
                "source_segment_index": cue.source_segment_index,
                "start_ms": cue.start_time,
                "end_ms": cue.end_time,
                "timing_source": cue.timing_source,
            }
            for cue in cues
            if getattr(cue, "needs_review", False)
        ],
    }
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
