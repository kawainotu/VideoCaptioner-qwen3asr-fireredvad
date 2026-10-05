"""Align supplied MiMo transcripts locally; never run local speech recognition."""

import argparse
import json
import sys


def align_manifest(manifest, model_dir, device="auto", on_segment=None):
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for local forced alignment")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    aligner = Qwen3ForcedAligner.from_pretrained(model_dir, dtype=dtype, device_map=device)
    audio, rate = sf.read(manifest["audio"], dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("Alignment input must be 16 kHz mono")
    output = []
    for index, segment in enumerate(manifest["segments"]):
        start, end = segment["start"], segment["end"]
        chunk = audio[round(start * rate / 1000) : round(end * rate / 1000)]
        if not len(chunk):
            raise ValueError("Empty alignment audio segment")
        language = manifest.get("language", "auto")
        if language == "auto":
            language = "zh" if any("\u3400" <= c <= "\u9fff" for c in segment["text"]) else "en"
        language = {"zh": "Chinese", "en": "English"}.get(language, language)
        result = aligner.align(audio=(chunk, rate), text=segment["text"], language=language)[0]
        output.append(
            {
                "text": segment["text"],
                "start": start,
                "end": end,
                "tokens": [
                    {
                        "text": item.text,
                        "start": float(item.start_time),
                        "end": float(item.end_time),
                    }
                    for item in result
                ],
            }
        )
        if on_segment:
            on_segment(index, output[-1])
        print(json.dumps({"progress": index + 1, "total": len(manifest["segments"])}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        result = align_manifest(manifest, args.model, args.device)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception as error:
        # Do not echo transcript, audio, credentials or model responses.
        sys.stderr.write(f"Forced alignment failed ({type(error).__name__})\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
