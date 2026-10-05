"""Isolated Silero / FireRed VAD runner script.

Executed via the isolated runtime python interpreter to run speech detection
without loading Qwen ASR or requiring any recognition models.
"""

import argparse
import gc
import json
import sys

# The runner is also executed directly by the isolated interpreter.
if __package__:
    from videocaptioner.core.mimo_vad_defaults import MIMO_FIRERED_VAD_DEFAULTS
else:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mimo_vad_defaults import MIMO_FIRERED_VAD_DEFAULTS

FIRERED_DEFAULTS = MIMO_FIRERED_VAD_DEFAULTS


def detect_regions(audio_path, vad_model="silero", model_dir=None, **options):
    """Load only the selected VAD and return seconds on the original audio timeline."""
    import soundfile as sf
    import torch

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    duration = len(audio) / 16000
    if vad_model == "firered":
        import numpy as np
        from fireredvad import FireRedVad, FireRedVadConfig

        config = FireRedVadConfig(
            use_gpu=False,
            **{
                key: options.get("firered_vad_" + key, default)
                for key, default in FIRERED_DEFAULTS.items()
            },
        )
        model = FireRedVad.from_pretrained(model_dir, config)
        pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        result, probabilities = model.detect((pcm, 16000))
        timestamps = result.get("timestamps", [])
        del probabilities, pcm
    elif vad_model == "silero":
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad(onnx=False)
        regions = get_speech_timestamps(
            torch.from_numpy(audio).float(),
            model,
            threshold=options.get("threshold", 0.5),
            sampling_rate=16000,
            min_speech_duration_ms=options.get("min_speech_ms", 250),
            min_silence_duration_ms=options.get("min_silence_ms", 500),
            speech_pad_ms=options.get("speech_pad_ms", 300),
            return_seconds=True,
        )
        timestamps = [(r["start"], r["end"]) for r in regions]
    else:
        raise ValueError(f"Unsupported VAD model: {vad_model}")
    results = []
    for start, end in timestamps:
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end > start:
            results.append({"start": round(start, 3), "end": round(end, 3)})
    del model, audio
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MiMo local VAD runner")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vad-model", choices=["silero", "firered"], default="silero")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=500)
    parser.add_argument("--speech-pad-ms", type=int, default=300)
    for key, default in FIRERED_DEFAULTS.items():
        parser.add_argument(
            "--firered-vad-" + key.replace("_", "-"),
            type=float if key == "speech_threshold" else int,
            default=default,
        )
    args = vars(parser.parse_args())
    output = args.pop("output")
    audio = args.pop("audio")
    try:
        results = detect_regions(audio, **args)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    except Exception as e:
        sys.stderr.write(f"VAD Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
