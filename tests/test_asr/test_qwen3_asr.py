from pathlib import Path
from types import SimpleNamespace

from videocaptioner.core.asr.asr_data import ASRDataSeg
from videocaptioner.core.asr.qwen3_asr import Qwen3ASR, group_word_segments
from videocaptioner.core.asr.qwen3_asr_runner import (
    AudioChunk,
    TranscribedChunk,
    approximate_segments,
    fixed_chunks,
    normalize_language,
    transcribe_chunks,
)
from videocaptioner.core.asr.qwen3_runtime import (
    is_model_ready,
    is_runtime_ready,
    write_runtime_marker,
)


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
