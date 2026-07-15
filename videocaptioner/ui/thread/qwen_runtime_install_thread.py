import os
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from videocaptioner.config import QWEN3_ASR_RUNTIME_PATH
from videocaptioner.core.asr.qwen3_runtime import (
    FIRERED_VAD_PACKAGE,
    QWEN_ASR_PACKAGE,
    SILERO_VAD_PACKAGE,
    is_supported_runtime_python_version,
    runtime_python_path,
    write_runtime_marker,
)

_SUPPORTED_PYTHON_TEXT = "Python 3.10, 3.11, or 3.12"
_PYPI_MIRROR_INDEX = "https://mirrors.aliyun.com/pypi/simple"
_PYPI_OFFICIAL_INDEX = "https://pypi.org/simple"
_PYTORCH_MIRROR_LINKS = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
_PYTORCH_OFFICIAL_INDEX = "https://download.pytorch.org/whl/cu128"
_TORCH_PACKAGE = "torch==2.11.0+cu128"
_TORCHAUDIO_PACKAGE = "torchaudio==2.11.0+cu128"


class QwenRuntimeInstallThread(QThread):
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, runtime_dir: Path = QWEN3_ASR_RUNTIME_PATH):
        super().__init__()
        self.runtime_dir = runtime_dir
        self.process: subprocess.Popen[str] | None = None
        self._stopped = False

    def _run_command(self, command: list[str], start: int, end: int, label: str) -> None:
        self.progress.emit(start, label)
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        assert self.process.stdout is not None
        lines = 0
        last_line = label
        for raw_line in self.process.stdout:
            if self._stopped:
                raise RuntimeError("Qwen3-ASR runtime installation cancelled")
            line = raw_line.strip()
            if not line:
                continue
            last_line = line[-160:]
            lines += 1
            progress = min(end - 1, start + lines // 4)
            self.progress.emit(progress, last_line)
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(last_line or f"Command failed with exit code {return_code}")
        self.progress.emit(end, label)

    def _run_command_with_fallback(
        self,
        primary_command: list[str],
        fallback_command: list[str],
        start: int,
        end: int,
        label: str,
    ) -> None:
        """Use the configured domestic mirror first, then retry via the official source."""
        try:
            self._run_command(primary_command, start, end, label)
        except RuntimeError as primary_error:
            if self._stopped or primary_command == fallback_command:
                raise
            fallback_label = self.tr("国内镜像不可用，正在切换官方源")
            self.progress.emit(start, fallback_label)
            try:
                self._run_command(fallback_command, start, end, label)
            except RuntimeError as fallback_error:
                raise RuntimeError(
                    f"{fallback_error} (domestic mirror also failed: {primary_error})"
                ) from fallback_error

    @staticmethod
    def _python_version(python: str) -> tuple[int, int] | None:
        """Read an interpreter version without importing its site packages."""
        try:
            result = subprocess.run(
                [
                    python,
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        try:
            major, minor = result.stdout.strip().splitlines()[-1].split(".", maxsplit=1)
            return int(major), int(minor)
        except (IndexError, ValueError):
            return None

    def _is_supported_python(self, python: str) -> bool:
        version = self._python_version(python)
        return version is not None and is_supported_runtime_python_version(version)

    @staticmethod
    def _launcher_python(launcher: str, version: str) -> str | None:
        """Resolve a registered Windows Python version through py.exe."""
        try:
            result = subprocess.run(
                [launcher, f"-{version}", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        candidate = result.stdout.strip().strip('"')
        return candidate if candidate and Path(candidate).is_file() else None

    def _find_supported_system_python(self) -> str | None:
        if os.name == "nt":
            launcher = shutil.which("py")
            if launcher:
                for version in ("3.12", "3.11", "3.10"):
                    candidate = self._launcher_python(launcher, version)
                    if candidate and self._is_supported_python(candidate):
                        return candidate

        for command in ("python3.12", "python3.11", "python3.10", "python"):
            candidate = shutil.which(command)
            if candidate and self._is_supported_python(candidate):
                return candidate
        return None

    def _bootstrap_python(self) -> str:
        if not getattr(sys, "frozen", False):
            if self._is_supported_python(sys.executable):
                return sys.executable
            raise RuntimeError(
                f"Qwen3-ASR with FireRedVAD requires {_SUPPORTED_PYTHON_TEXT}; "
                f"the current interpreter is Python {sys.version_info.major}.{sys.version_info.minor}."
            )

        configured = os.environ.get("VIDEOCAPTIONER_QWEN_PYTHON")
        if configured:
            if not Path(configured).is_file():
                raise RuntimeError(
                    "VIDEOCAPTIONER_QWEN_PYTHON does not point to a Python executable: "
                    f"{configured}"
                )
            if self._is_supported_python(configured):
                return configured
            version = self._python_version(configured)
            version_text = f"{version[0]}.{version[1]}" if version else "unknown"
            raise RuntimeError(
                f"VIDEOCAPTIONER_QWEN_PYTHON uses Python {version_text}; "
                f"Qwen3-ASR with FireRedVAD requires {_SUPPORTED_PYTHON_TEXT}."
            )

        system_python = self._find_supported_system_python()
        if system_python:
            return system_python
        raise RuntimeError(
            f"Qwen3-ASR with FireRedVAD requires {_SUPPORTED_PYTHON_TEXT}. "
            "Install one of these versions or set VIDEOCAPTIONER_QWEN_PYTHON."
        )

    def _ensure_runtime_python(self) -> str:
        runtime_python = runtime_python_path(self.runtime_dir)
        if runtime_python.is_file() and self._is_supported_python(str(runtime_python)):
            return str(runtime_python)

        if self.runtime_dir.exists():
            self.progress.emit(0, "Rebuilding incompatible Qwen3-ASR runtime")
            try:
                shutil.rmtree(self.runtime_dir)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not recreate the Qwen3-ASR runtime at {self.runtime_dir}: {exc}"
                ) from exc

        bootstrap_python = self._bootstrap_python()
        self._run_command(
            [bootstrap_python, "-m", "venv", str(self.runtime_dir)],
            0,
            10,
            "Creating isolated runtime",
        )
        return str(runtime_python_path(self.runtime_dir))

    def run(self) -> None:
        try:
            self.runtime_dir.parent.mkdir(parents=True, exist_ok=True)
            python = self._ensure_runtime_python()
            pypi_index = os.environ.get("VIDEOCAPTIONER_PYPI_INDEX", _PYPI_MIRROR_INDEX)
            pytorch_mirror = os.environ.get(
                "VIDEOCAPTIONER_PYTORCH_MIRROR", _PYTORCH_MIRROR_LINKS
            )
            self._run_command_with_fallback(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                    "wheel",
                    "--index-url",
                    pypi_index,
                ],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                    "wheel",
                    "--index-url",
                    _PYPI_OFFICIAL_INDEX,
                ],
                10,
                20,
                self.tr("正在更新安装工具"),
            )
            self._run_command_with_fallback(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    _TORCH_PACKAGE,
                    _TORCHAUDIO_PACKAGE,
                    "--index-url",
                    pypi_index,
                    "--find-links",
                    pytorch_mirror,
                ],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    _TORCH_PACKAGE,
                    _TORCHAUDIO_PACKAGE,
                    "--index-url",
                    _PYTORCH_OFFICIAL_INDEX,
                ],
                20,
                65,
                self.tr("正在安装 PyTorch CUDA 运行库"),
            )
            self._run_command_with_fallback(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    QWEN_ASR_PACKAGE,
                    SILERO_VAD_PACKAGE,
                    "--index-url",
                    pypi_index,
                ],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    QWEN_ASR_PACKAGE,
                    SILERO_VAD_PACKAGE,
                    "--index-url",
                    _PYPI_OFFICIAL_INDEX,
                ],
                65,
                88,
                self.tr("正在安装 Qwen3-ASR 和 Silero VAD"),
            )
            self._run_command_with_fallback(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    FIRERED_VAD_PACKAGE,
                    "--index-url",
                    pypi_index,
                ],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    FIRERED_VAD_PACKAGE,
                    "--index-url",
                    _PYPI_OFFICIAL_INDEX,
                ],
                88,
                95,
                self.tr("正在安装 FireRedVAD"),
            )
            self._run_command(
                [
                    python,
                    "-c",
                    "import fireredvad, kaldi_native_fbank, qwen_asr, silero_vad, torch; "
                    "print(torch.__version__)",
                ],
                95,
                99,
                self.tr("正在验证运行环境"),
            )
            write_runtime_marker(self.runtime_dir)
            self.progress.emit(100, self.tr("Qwen3-ASR 运行环境安装完成"))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.process = None

    def stop(self) -> None:
        self._stopped = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
