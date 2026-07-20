from pathlib import Path
from types import SimpleNamespace

from videocaptioner.core.entities import (
    TranscribeLanguageEnum,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
)
from videocaptioner.ui import task_factory


class _ConfigValue:
    def __init__(self, value):
        self.value = value


class _ConfigStub:
    def __init__(self, model, word_timestamps, need_split):
        self.transcribe_model = _ConfigValue(model)
        self.qwen_asr_word_timestamps = _ConfigValue(word_timestamps)
        self.need_split = _ConfigValue(need_split)
        self.work_dir = _ConfigValue(Path("output"))
        self.transcribe_language = _ConfigValue(TranscribeLanguageEnum.AUTO)
        self.transcribe_output_format = _ConfigValue(TranscribeOutputFormatEnum.SRT)

    def __getattr__(self, _name):
        return _ConfigValue("")


def _patch_task_factory(monkeypatch, config):
    monkeypatch.setattr(task_factory, "cfg", config)
    monkeypatch.setattr(task_factory, "runtime_python_path", lambda: Path("runtime/python.exe"))
    monkeypatch.setattr(
        task_factory,
        "get_qwen3_asr_model",
        lambda _key: SimpleNamespace(path=Path("models/qwen3-asr")),
    )
    monkeypatch.setattr(
        task_factory,
        "resolve_firered_vad_model_path",
        lambda: Path("models/firered-vad"),
    )


def test_standalone_qwen_transcription_uses_word_timestamps_when_enabled(monkeypatch):
    _patch_task_factory(
        monkeypatch,
        _ConfigStub(
            TranscribeModelEnum.QWEN3_ASR,
            word_timestamps=True,
            need_split=False,
        ),
    )

    task = task_factory.TaskFactory.create_transcribe_task("C:/media/sample.mp4")

    assert task.transcribe_config.need_word_time_stamp is True
    assert Path(task.output_path) == Path("C:/media/sample.srt")


def test_standalone_qwen_transcription_keeps_segment_timestamps_when_disabled(monkeypatch):
    _patch_task_factory(
        monkeypatch,
        _ConfigStub(
            TranscribeModelEnum.QWEN3_ASR,
            word_timestamps=False,
            need_split=True,
        ),
    )

    task = task_factory.TaskFactory.create_transcribe_task("C:/media/sample.mp4")

    assert task.transcribe_config.need_word_time_stamp is False


def test_full_pipeline_still_uses_subtitle_split_setting(monkeypatch):
    _patch_task_factory(
        monkeypatch,
        _ConfigStub(
            TranscribeModelEnum.QWEN3_ASR,
            word_timestamps=False,
            need_split=True,
        ),
    )

    task = task_factory.TaskFactory.create_transcribe_task(
        "C:/media/sample.mp4", need_next_task=True
    )

    assert task.transcribe_config.need_word_time_stamp is True
