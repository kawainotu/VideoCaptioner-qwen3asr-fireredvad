"""Validate a downloaded Qwen result and export subtitles without any network request."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, help="Raw response or saved failure JSON")
    parser.add_argument("--srt", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--audio", help="Optional original audio, for duration validation")
    parser.add_argument("--sentence-mode", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from videocaptioner.core.asr.asr_data import ASRData
    from videocaptioner.core.asr.qwen_filetrans_asr import QwenFileTransASR, _comparable_text

    source = Path(args.response)
    output, summary_path = Path(args.srt), Path(args.summary)
    for path in (output, summary_path):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    if output.resolve() == summary_path.resolve():
        parser.error("SRT and summary must have different paths")
    saved = json.loads(source.read_text(encoding="utf-8-sig"))
    wrapped = isinstance(saved, dict) and saved.get("format_version") == 1 and "response" in saved
    data = saved["response"] if wrapped else saved
    runner = QwenFileTransASR(
        args.audio or source,
        "offline-replay",
        need_word_time_stamp=not args.sentence_mode,
    )
    if args.audio:
        runner._preflight_audio()
    elif wrapped:
        duration = saved.get("audio_duration_ms")
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0
        ):
            parser.error("Saved audio duration must be a positive integer")
        runner._audio_duration_ms = duration
    cues = runner._make_segments(data)
    result = ASRData(cues)
    original_text = "".join(s["text"] for t in data["transcripts"] for s in t["sentences"])
    if _comparable_text(original_text) != _comparable_text(
        "".join(s.text for s in result.segments)
    ):
        raise RuntimeError("Export would change or reorder recognized text")
    summary = {
        "source_sentences": sum(len(t["sentences"]) for t in data["transcripts"]),
        "subtitle_cues": len(result.segments),
        "text_preserved": True,
        "audio_duration_checked": runner._audio_duration_ms is not None,
        "nonoverlapping": all(
            a.end_time <= b.start_time for a, b in zip(result.segments, result.segments[1:])
        ),
        "needs_review": [
            {
                "source_sentence": cue.source_segment_index,
                "start_ms": cue.start_time,
                "end_ms": cue.end_time,
                "timing_source": cue.timing_source,
                "reason": cue.timing_review_reason,
            }
            for cue in result.segments
            if getattr(cue, "needs_review", False)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
