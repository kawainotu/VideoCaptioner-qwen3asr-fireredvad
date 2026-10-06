import importlib
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from videocaptioner.cli.config import DEFAULTS, build_config, load_config_file, save_config_value
from videocaptioner.cli.main import _build_cli_overrides, build_parser
from videocaptioner.cli.validators import validate_transcribe
from videocaptioner.core.asr.qwen_filetrans_asr import DEFAULT_MODEL, QwenFileTransASR
from videocaptioner.core.entities import (
    LANGUAGES,
    TranscribeConfig,
    TranscribeLanguageEnum,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
    TranscribeTask,
    get_asr_language_capability,
)


@pytest.mark.parametrize("command", ["transcribe", "process"])
def test_cli_cloud_options_flow_to_config(command):
    args = build_parser().parse_args(
        [
            command,
            "audio.wav",
            "--asr",
            "qwen-filetrans",
            "--qwen-filetrans-key",
            "test-key",
            "--qwen-filetrans-base",
            "https://proxy/api/v1",
            "--qwen-filetrans-model",
            DEFAULT_MODEL,
        ]
    )
    overrides = _build_cli_overrides(args)
    assert overrides["transcribe"]["asr"] == "qwen-filetrans"
    assert overrides["qwen_filetrans"] == {
        "api_key": "test-key",
        "api_base": "https://proxy/api/v1",
        "model": DEFAULT_MODEL,
    }


def test_config_env_priority_roundtrip_and_validation(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    save_config_value("qwen_filetrans.api_key", "from-file", config_path=path)
    save_config_value("qwen_filetrans.task_timeout", "3600", config_path=path)
    assert load_config_file(path)["qwen_filetrans"]["task_timeout"] == 3600
    monkeypatch.setenv("VIDEOCAPTIONER_QWEN_FILETRANS_API_KEY", "from-env")
    config = build_config(config_path=path, cli_overrides={"transcribe": {"asr": "qwen-filetrans"}})
    assert config["qwen_filetrans"]["api_key"] == "from-env"
    assert validate_transcribe(config)
    config["transcribe"]["language"] = "yue"
    assert not validate_transcribe(config)
    config["transcribe"]["language"] = "auto"
    config["qwen_filetrans"]["api_key"] = ""
    assert not validate_transcribe(config)


def test_cloud_factory_needs_no_local_models_and_preserves_native_timing(monkeypatch, tmp_path):
    module = importlib.import_module("videocaptioner.core.asr.transcribe")
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.QWEN_FILETRANS,
        qwen_filetrans_api_key="test-key",
        need_word_time_stamp=False,
        qwen_filetrans_hotwords="斯蒂格勒",
        qwen_filetrans_vocabulary_id="vocab-existing",
        qwen_filetrans_context="录音讨论第三持存。",
    )
    instance = module._create_asr_instance(str(tmp_path / "audio.wav"), config)
    assert isinstance(instance, QwenFileTransASR)
    assert instance.model == DEFAULT_MODEL
    assert instance.vocabulary == {"斯蒂格勒": 4}
    assert instance.vocabulary_id == "vocab-existing"
    assert instance.context == "录音讨论第三持存。"
    result = Mock()
    monkeypatch.setattr(instance, "run", Mock(return_value=result))
    monkeypatch.setattr(module, "_create_asr_instance", lambda *_: instance)
    assert module.transcribe("audio.wav", config) is result
    result.optimize_timing.assert_not_called()


@pytest.mark.parametrize("word_mode,pipeline", [(True, False), (False, False), (False, True)])
def test_gui_task_factory_carries_cloud_settings(word_mode, pipeline, monkeypatch):
    from videocaptioner.ui import task_factory

    class ConfigStub:
        values = {
            "transcribe_model": TranscribeModelEnum.QWEN_FILETRANS,
            "transcribe_language": TranscribeLanguageEnum.AUTO,
            "transcribe_output_format": TranscribeOutputFormatEnum.SRT,
            "qwen_filetrans_word_timestamps": word_mode,
            "need_split": True,
            "qwen_filetrans_api_key": "test-key",
            "qwen_filetrans_api_base": "https://proxy/api/v1",
            "qwen_filetrans_model": DEFAULT_MODEL,
            "qwen_filetrans_hotwords": "斯蒂格勒\n第三持存 | 5",
            "qwen_filetrans_vocabulary_id": "vocab-existing",
            "qwen_filetrans_context": "录音讨论第三持存。",
            "work_dir": "output",
        }

        def __getattr__(self, key):
            return SimpleNamespace(value=self.values.get(key, ""))

    monkeypatch.setattr(task_factory, "cfg", ConfigStub())
    task = task_factory.TaskFactory.create_transcribe_task(
        "C:/media/video.mp4", need_next_task=pipeline
    )
    config = task.transcribe_config
    assert config.transcribe_model == TranscribeModelEnum.QWEN_FILETRANS
    assert config.need_word_time_stamp is (word_mode or pipeline)
    assert config.qwen_filetrans_api_key == "test-key"
    assert config.qwen_filetrans_api_base == "https://proxy/api/v1"
    assert config.qwen_filetrans_model == DEFAULT_MODEL
    assert config.qwen_filetrans_hotwords == "斯蒂格勒\n第三持存 | 5"
    assert config.qwen_filetrans_vocabulary_id == "vocab-existing"
    assert config.qwen_filetrans_context == "录音讨论第三持存。"
    assert "test-key" not in config.print_config()
    assert "斯蒂格勒" not in config.print_config() and "录音讨论" not in config.print_config()


def test_language_capability_uses_supported_hints_only():
    capability = get_asr_language_capability(TranscribeModelEnum.QWEN_FILETRANS)
    assert capability.supports_auto
    codes = {LANGUAGES[language.value] for language in capability.supported_languages}
    assert "zh" in codes and "en" in codes and "yue" not in codes


def test_shutdown_and_thread_cancel_reaches_cloud_asr(monkeypatch):
    from videocaptioner.ui.thread import transcript_thread as module

    thread = module.TranscriptThread(
        TranscribeTask(
            transcribe_config=TranscribeConfig(transcribe_model=TranscribeModelEnum.QWEN_FILETRANS)
        )
    )
    asr = Mock()
    thread._on_asr_created(asr)
    monkeypatch.setattr(module, "_ACTIVE_TRANSCRIPT_THREADS", {thread})
    module.cancel_active_mimo_transcriptions()
    asr.cancel.assert_called_once()
    assert thread._cancelled.is_set()


def test_connection_probe_only_requests_policy_and_ignores_cancelled_result(monkeypatch):
    from videocaptioner.ui.components import QwenFileTransSettingWidget as module

    adapter = Mock()
    monkeypatch.setattr(module, "QwenFileTransASR", lambda *_: adapter)
    worker = module.QwenFileTransConnectionThread("https://proxy/api/v1", "test-key", DEFAULT_MODEL)
    result = Mock()
    worker.result_ready.connect(result)
    worker.run()
    adapter.get_upload_policy.assert_called_once()
    adapter.run.assert_not_called()
    result.assert_called_once()
    worker = module.QwenFileTransConnectionThread("https://proxy/api/v1", "test-key", DEFAULT_MODEL)
    result = Mock()
    worker.result_ready.connect(result)
    adapter.get_upload_policy.side_effect = lambda *_: worker.stop()
    worker.run()
    result.assert_not_called()
    adapter.cancel.assert_called_once()


def test_connection_worker_registry_and_shutdown_are_retained_until_finished(monkeypatch):
    from videocaptioner.ui.components import QwenFileTransSettingWidget as module

    app = Mock()
    monkeypatch.setattr(module.QCoreApplication, "instance", lambda: app)
    monkeypatch.setattr(module.QThread, "start", lambda *_: None)
    monkeypatch.setattr(module, "_LAST_APP", None)
    worker = module.QwenFileTransConnectionThread("https://proxy/api/v1", "test-key", DEFAULT_MODEL)
    worker.start()
    assert worker in module._ACTIVE_QWEN_WORKERS
    app.aboutToQuit.connect.assert_called_once_with(module._stop_workers)
    module._stop_workers()
    assert worker._stopped and worker in module._ACTIVE_QWEN_WORKERS
    worker._cleanup()
    assert worker not in module._ACTIVE_QWEN_WORKERS


def test_cli_transcribe_passes_cloud_provider_through_to_core(tmp_path, monkeypatch):
    import videocaptioner.core.asr as asr_module
    from videocaptioner.cli.commands import transcribe as command

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    seen = []

    def transcribe(path, config, callback):
        seen.append(config)
        return Mock()

    monkeypatch.setattr(asr_module, "transcribe", transcribe)
    config = deepcopy(DEFAULTS)
    config["transcribe"]["asr"] = "qwen-filetrans"
    config["qwen_filetrans"]["api_key"] = "test-key"
    config["qwen_filetrans"]["hotwords"] = "斯蒂格勒 | 4"
    config["qwen_filetrans"]["vocabulary_id"] = "vocab-existing"
    config["qwen_filetrans"]["context"] = "录音讨论第三持存。"
    args = Namespace(
        input=str(audio),
        output=str(tmp_path / "output.srt"),
        word_timestamps=True,
        verbose=False,
        quiet=True,
    )
    assert command.run(args, config) == 0
    assert seen[0].qwen_filetrans_api_key == "test-key"
    assert seen[0].transcribe_model == TranscribeModelEnum.QWEN_FILETRANS
    assert seen[0].need_word_time_stamp
    assert seen[0].qwen_filetrans_hotwords == "斯蒂格勒 | 4"
    assert seen[0].qwen_filetrans_vocabulary_id == "vocab-existing"
    assert seen[0].qwen_filetrans_context == "录音讨论第三持存。"


@pytest.mark.parametrize("command", ["transcribe", "process"])
def test_cli_enhancement_options_and_utf8_bom_file(command, tmp_path):
    path = tmp_path / "热词.txt"
    path.write_text("斯蒂格勒\n第三持存 | 5", encoding="utf-8-sig")
    args = build_parser().parse_args(
        [
            command,
            "audio.wav",
            "--asr",
            "qwen-filetrans",
            "--qwen-filetrans-hotwords-file",
            str(path),
            "--qwen-filetrans-vocabulary-id",
            "vocab-existing",
            "--qwen-filetrans-context",
            "录音讨论第三持存。",
        ]
    )
    overrides = _build_cli_overrides(args)["qwen_filetrans"]
    assert overrides == {
        "hotwords": "斯蒂格勒\n第三持存 | 5",
        "vocabulary_id": "vocab-existing",
        "context": "录音讨论第三持存。",
    }


def test_cli_enhancements_toml_roundtrip_and_early_validation(tmp_path):
    path = tmp_path / "config.toml"
    save_config_value("qwen_filetrans.hotwords", "斯蒂格勒\n第三持存 | 5", config_path=path)
    save_config_value("qwen_filetrans.context", "录音讨论第三持存。", config_path=path)
    loaded = load_config_file(path)
    assert loaded["qwen_filetrans"]["hotwords"] == "斯蒂格勒\n第三持存 | 5"
    assert loaded["qwen_filetrans"]["context"] == "录音讨论第三持存。"
    config = deepcopy(DEFAULTS)
    config["transcribe"]["asr"] = "qwen-filetrans"
    config["qwen_filetrans"]["api_key"] = "test-key"
    config["qwen_filetrans"]["hotwords"] = "词条 | 6"
    assert not validate_transcribe(config)


def test_hotword_file_error_is_clear_and_does_not_echo_private_contents(tmp_path):
    args = build_parser().parse_args(
        [
            "transcribe",
            "audio.wav",
            "--asr",
            "qwen-filetrans",
            "--qwen-filetrans-hotwords-file",
            str(tmp_path / "missing.txt"),
        ]
    )
    with pytest.raises(ValueError, match="UTF-8"):
        _build_cli_overrides(args)
