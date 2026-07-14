import os
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from videocaptioner.config import QWEN3_ASR_RUNTIME_PATH
from videocaptioner.core.asr.qwen3_runtime import (
    QWEN_ASR_PACKAGE,
    SILERO_VAD_PACKAGE,
    runtime_python_path,
    write_runtime_marker,
)


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

    def _bootstrap_python(self) -> str:
        if not getattr(sys, "frozen", False):
            return sys.executable
        configured = os.environ.get("VIDEOCAPTIONER_QWEN_PYTHON")
        if configured and Path(configured).is_file():
            return configured
        system_python = shutil.which("python")
        if system_python:
            return system_python
        raise RuntimeError(
            "A Python interpreter is required to install the Qwen3-ASR runtime. "
            "Install Python 3.12 or set VIDEOCAPTIONER_QWEN_PYTHON."
        )

    def run(self) -> None:
        try:
            self.runtime_dir.parent.mkdir(parents=True, exist_ok=True)
            if not runtime_python_path(self.runtime_dir).is_file():
                bootstrap_python = self._bootstrap_python()
                self._run_command(
                    [bootstrap_python, "-m", "venv", str(self.runtime_dir)],
                    0,
                    10,
                    self.tr("正在创建独立运行环境"),
                )

            python = str(runtime_python_path(self.runtime_dir))
            self._run_command(
                [python, "-m", "pip", "install", "--upgrade", "pip", "wheel"],
                10,
                20,
                self.tr("正在更新安装工具"),
            )
            self._run_command(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
                20,
                65,
                self.tr("正在安装 PyTorch CUDA 运行库"),
            )
            self._run_command(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    QWEN_ASR_PACKAGE,
                    SILERO_VAD_PACKAGE,
                ],
                65,
                95,
                self.tr("正在安装 Qwen3-ASR 与 Silero VAD"),
            )
            self._run_command(
                [
                    python,
                    "-c",
                    "import qwen_asr, silero_vad, torch; print(torch.__version__)",
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
