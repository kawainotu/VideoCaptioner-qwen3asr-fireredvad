import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from videocaptioner.core.asr.asr_data import ASRDataSeg
from videocaptioner.core.asr.qwen3_asr import Qwen3ASR, group_word_segments
from videocaptioner.core.asr.qwen3_asr_runner import (
    AudioChunk,
    TranscribedChunk,
    approximate_segments,
    build_parser,
    firered_timestamps_to_regions,
    fixed_chunks,
    normalize_language,
    transcribe_chunks,
)
from videocaptioner.core.asr.qwen3_models import (
    DEFAULT_QWEN3_ASR_MODEL_KEY,
    get_qwen3_asr_model,
)
from videocaptioner.core.asr.qwen3_runtime import (
    is_model_ready,
    is_runtime_ready,
    write_runtime_marker,
)
from videocaptioner.core.asr.qwen3_vad_models import (
    FIRERED_VAD_MODEL_KEY,
    get_qwen3_vad_model,
    is_firered_vad_model_ready,
)
from videocaptioner.core.qwen3_vad_defaults import FIRERED_VAD_DEFAULTS


class TestQwenSegmentGrouping:
    def test_groups_cjk_until_sentence_end(self):
        source = [
            ASRDataSeg("你", 0, 100),
            ASRDataSeg("好", 100, 200),
            ASRDataSeg("。", 200, 300),
            ASRDataSeg("再", 400, 500),
            ASRDataSeg("见", 500, 600),
        ]

        grouped = group_word_segments(source)

        assert [(segment.text, segment.start_time, segment.end_time) for segment in grouped] == [
            ("你好。", 0, 300),
            ("再见", 400, 600),
        ]

    def test_groups_english_with_spaces(self):
        source = [
            ASRDataSeg("Hello", 0, 200),
            ASRDataSeg("world", 200, 400),
            ASRDataSeg("!", 400, 450),
        ]

        grouped = group_word_segments(source)

        assert grouped[0].text == "Hello world!"

    def test_keeps_word_segments_for_pipeline(self):
        asr = Qwen3ASR.__new__(Qwen3ASR)
        asr.need_word_time_stamp = True
        payload = {
            "segments": [
                {"text": "你", "start_time": 0, "end_time": 100},
                {"text": "好", "start_time": 100, "end_time": 200},
            ]
        }

        result = asr._make_segments(payload)

        assert [segment.text for segment in result] == ["你", "好"]


class TestQwenRunnerHelpers:
    def test_runner_drains_diagnostics_without_blocking_progress(self, tmp_path):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"audio")
        output_payload = '{"segments": [], "language": ""}'
        runner_script = (
            "import json, pathlib, sys;"
            "sys.stderr.write('x' * 131072 + '\\n');"
            "sys.stderr.flush();"
            "print(json.dumps({'type': 'progress', 'progress': 42, "
            "'message': 'Transcribed 1/1'}), flush=True);"
            "pathlib.Path(sys.argv[1]).write_text(" + repr(output_payload) + ", "
            "encoding='utf-8')"
        )

        asr = Qwen3ASR.__new__(Qwen3ASR)
        asr.audio_input = str(audio_path)
        asr.file_binary = None
        asr.process = None
        asr._validate_components = lambda: None
        asr._build_command = lambda _audio, output: [
            sys.executable,
            "-c",
            runner_script,
            str(output),
        ]
        events = []

        result = asr._run(callback=lambda value, message: events.append((value, message)))

        assert result == {"segments": [], "language": ""}
        assert (42, "Transcribed 1/1") in events
        assert asr.process is None

    def test_transcription_uses_qwen_language_names(self):
        received_languages = []

        class Model:
            def transcribe(self, **kwargs):
                received_languages.append(kwargs["language"])
                return [SimpleNamespace(text="test", language="Japanese")]

        chunks = [AudioChunk(audio=[0.0] * 16000, offset_seconds=0)]
        args = SimpleNamespace(context="", language="ja")

        transcribe_chunks(chunks, Model(), args)

        assert received_languages == ["Japanese"]

    def test_normalizes_supported_language_codes(self):
        assert normalize_language("zh") == "Chinese"
        assert normalize_language("ja") == "Japanese"
        assert normalize_language("pt") == "Portuguese"
        assert normalize_language("English") == "English"
        assert normalize_language("Chinese,English") == "Chinese"

    def test_approximates_unsupported_language_timestamps(self):
        chunk = TranscribedChunk(
            audio=[0.0] * 16000,
            offset_seconds=2.0,
            text="merhaba dunya",
            language="Turkish",
        )

        result = approximate_segments(chunk)

        assert [item["text"] for item in result] == ["merhaba", "dunya"]
        assert result[0]["start_time"] == 2000
        assert result[-1]["end_time"] == 3000

    def test_splits_long_audio_into_safe_chunks(self):
        audio = [0.0] * (171 * 16000)

        chunks = fixed_chunks(audio)

        assert len(chunks) == 2
        assert isinstance(chunks[0], AudioChunk)
        assert chunks[0].offset_seconds == 0
        assert chunks[1].offset_seconds == 170

    def test_converts_firered_timestamps_to_sample_regions(self):
        regions = firered_timestamps_to_regions([(0.25, 1.5)], 32000)

        assert regions == [{"start": 4000, "end": 24000}]

    def test_builds_firered_runner_arguments(self, tmp_path):
        model_dir = tmp_path / "FireRedVAD" / "VAD"
        asr = Qwen3ASR(
            b"",
            vad_model=FIRERED_VAD_MODEL_KEY,
            firered_vad_model_dir=str(model_dir),
            firered_vad_max_speech_frame=2400,
        )

        command = asr._build_command(tmp_path / "audio.wav", tmp_path / "result.json")

        assert command[command.index("--vad-model") + 1] == FIRERED_VAD_MODEL_KEY
        assert command[command.index("--firered-vad-model") + 1] == str(model_dir)
        assert command[command.index("--firered-vad-max-speech-frame") + 1] == "2400"

    def test_firered_defaults_stay_in_sync_with_isolated_runner(self):
        expected = asdict(FIRERED_VAD_DEFAULTS)
        asr = Qwen3ASR(b"")
        args = build_parser().parse_args(
            [
                "--audio",
                "audio.wav",
                "--output",
                "result.json",
                "--asr-model",
                "asr",
                "--aligner-model",
                "aligner",
            ]
        )

        for name, value in expected.items():
            assert getattr(asr, f"firered_vad_{name}") == value
            assert getattr(args, f"firered_vad_{name}") == value


class TestQwenRuntimeChecks:
    def test_runtime_marker_requires_python_and_matching_version(self, tmp_path):
        runtime = tmp_path / "runtime"
        python = runtime / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")

        write_runtime_marker(runtime)

        assert is_runtime_ready(runtime)

    def test_model_ready_requires_config_and_weights(self, tmp_path):
        model_dir = Path(tmp_path) / "model"
        model_dir.mkdir()
        assert not is_model_ready(model_dir)

        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        assert not is_model_ready(model_dir)

        (model_dir / "model.safetensors").write_bytes(b"weights")
        assert is_model_ready(model_dir)

    def test_firered_model_ready_requires_cmvn_and_weights(self, tmp_path):
        model_dir = Path(tmp_path) / "FireRedVAD" / "VAD"
        model_dir.mkdir(parents=True)
        (model_dir / "cmvn.ark").write_bytes(b"cmvn")
        assert not is_firered_vad_model_ready(model_dir)

        (model_dir / "model.pth.tar").write_bytes(b"weights")
        assert is_firered_vad_model_ready(model_dir)


class TestQwenModelVariants:
    def test_uses_the_official_model_when_selection_is_missing(self):
        model = get_qwen3_asr_model(None)

        assert model.key == DEFAULT_QWEN3_ASR_MODEL_KEY
        assert model.model_id == "Qwen/Qwen3-ASR-1.7B"
        assert model.path.name == "Qwen3-ASR-1.7B"

    def test_resolves_the_japanese_anime_galgame_variant(self):
        model = get_qwen3_asr_model("qwen3-asr-1.7b-ja-anime-galgame")

        assert model.model_id == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
        assert model.source == "huggingface"
        assert model.path.name == "Qwen3-ASR-1.7B-JA-Anime-Galgame"
        assert "optimizer.pt" in model.ignore_patterns

    def test_resolves_firered_vad_choice(self):
        model = get_qwen3_vad_model(FIRERED_VAD_MODEL_KEY)

        assert model.model_id == "FireRedTeam/FireRedVAD"
        assert model.path is not None
        assert model.path.name == "FireRedVAD"
