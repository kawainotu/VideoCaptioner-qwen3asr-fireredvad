import sys

import pytest

import videocaptioner.ui.thread.qwen_runtime_install_thread as runtime_install_module
from videocaptioner.ui.thread.qwen_runtime_install_thread import QwenRuntimeInstallThread


def test_frozen_bootstrap_uses_supported_system_python(monkeypatch, tmp_path):
    runtime = QwenRuntimeInstallThread(tmp_path / "runtime")
    supported_python = str(tmp_path / "python.exe")

    monkeypatch.setattr(runtime_install_module.sys, "frozen", True, raising=False)
    monkeypatch.delenv("VIDEOCAPTIONER_QWEN_PYTHON", raising=False)
    monkeypatch.setattr(runtime, "_find_supported_system_python", lambda: supported_python)

    assert runtime._bootstrap_python() == supported_python


def test_frozen_bootstrap_rejects_python_314_from_override(monkeypatch, tmp_path):
    runtime = QwenRuntimeInstallThread(tmp_path / "runtime")
    python = tmp_path / "python.exe"
    python.touch()

    monkeypatch.setattr(runtime_install_module.sys, "frozen", True, raising=False)
    monkeypatch.setenv("VIDEOCAPTIONER_QWEN_PYTHON", str(python))
    monkeypatch.setattr(runtime, "_python_version", lambda _: (3, 14))

    with pytest.raises(RuntimeError, match="Python 3.10, 3.11, or 3.12"):
        runtime._bootstrap_python()


def test_non_frozen_bootstrap_rejects_an_unsupported_interpreter(monkeypatch, tmp_path):
    runtime = QwenRuntimeInstallThread(tmp_path / "runtime")

    monkeypatch.delattr(runtime_install_module.sys, "frozen", raising=False)
    monkeypatch.setattr(runtime, "_python_version", lambda _: (3, 14))

    with pytest.raises(RuntimeError, match=f"Python {sys.version_info.major}.{sys.version_info.minor}"):
        runtime._bootstrap_python()


def test_ensure_runtime_rebuilds_an_unsupported_existing_environment(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime"
    existing_python = runtime_dir / "Scripts" / "python.exe"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    runtime = QwenRuntimeInstallThread(runtime_dir)
    commands = []

    monkeypatch.setattr(runtime, "_is_supported_python", lambda _: False)
    monkeypatch.setattr(runtime, "_bootstrap_python", lambda: "bootstrap-python")
    monkeypatch.setattr(runtime, "_run_command", lambda *args: commands.append(args))

    assert runtime._ensure_runtime_python() == str(existing_python)
    assert not runtime_dir.exists()
    assert commands == [
        (
            ["bootstrap-python", "-m", "venv", str(runtime_dir)],
            0,
            10,
            "Creating isolated runtime",
        )
    ]


def test_command_falls_back_to_official_source(monkeypatch, tmp_path):
    runtime = QwenRuntimeInstallThread(tmp_path / "runtime")
    commands = []
    progress = []

    def run_command(command, *args):
        commands.append(command)
        if command == ["mirror"]:
            raise RuntimeError("mirror unavailable")

    monkeypatch.setattr(runtime, "_run_command", run_command)
    runtime.progress.connect(lambda value, message: progress.append((value, message)))

    runtime._run_command_with_fallback(["mirror"], ["official"], 20, 65, "install")

    assert commands == [["mirror"], ["official"]]
    assert any("官方源" in message for _, message in progress)
