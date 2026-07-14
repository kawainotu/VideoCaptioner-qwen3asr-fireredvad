import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Union

from videocaptioner.config import (
    QWEN3_ALIGNER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
)

from ..utils.logger import setup_logger
from .asr_data import ASRDataSeg
from .base import BaseASR
from .qwen3_runtime import missing_components, runtime_python_path

logger = setup_logger("qwen3_asr")

_SENTENCE_END_RE = re.compile(r"[.!?。！？…][\"'”’）】》」』]*$")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_NO_SPACE_BEFORE_RE = re.compile(r"^[,.;:!?，。；：！？、…\)\]】》」』”’]")


def _join_token_text(current: str, token: str) -> str:
    token = token.strip()
    if not current or not token:
        return current + token
    if _CJK_RE.search(current[-1]) or _CJK_RE.search(token[0]):
        return current + token
    if _NO_SPACE_BEFORE_RE.search(token):
        return current + token
    return current + " " + token


def group_word_segments(
    segments: list[ASRDataSeg],
    max_chars: int = 42,
    max_duration_ms: int = 8000,
    pause_ms: int = 700,
) -> list[ASRDataSeg]:
    if not segments:
        return []

    grouped: list[ASRDataSeg] = []
    text = ""
    start = segments[0].start_time
    end = segments[0].end_time

    for index, segment in enumerate(segments):
        if text and segment.start_time - end >= pause_ms:
            grouped.append(ASRDataSeg(text.strip(), start, end))
            text = ""
            start = segment.start_time

        text = _join_token_text(text, segment.text)
        end = segment.end_time
        next_gap = 0
        if index + 1 < len(segments):
            next_gap = segments[index + 1].start_time - end

        should_close = (
            bool(_SENTENCE_END_RE.search(text))
            or len(text) >= max_chars
            or end - start >= max_duration_ms
            or next_gap >= pause_ms
        )
        if should_close:
            grouped.append(ASRDataSeg(text.strip(), start, end))
            text = ""
            if index + 1 < len(segments):
                start = segments[index + 1].start_time

    if text:
        grouped.append(ASRDataSeg(text.strip(), start, end))
    return grouped


class Qwen3ASR(BaseASR):
    """Qwen3-ASR adapter backed by an isolated Python runner."""

    def __init__(
        self,
        audio_input: Union[str, bytes],
        runtime_python: Optional[str] = None,
        asr_model_dir: Optional[str] = None,
        aligner_model_dir: Optional[str] = None,
        language: str = "",
        device: str = "auto",
        low_memory: bool = True,
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
        vad_min_speech_ms: int = 250,
        vad_min_silence_ms: int = 500,
        vad_speech_pad_ms: int = 300,
        prompt: str = "",
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
    ):
        super().__init__(audio_input, use_cache, need_word_time_stamp)
        self.runtime_python = Path(runtime_python) if runtime_python else runtime_python_path()
        self.asr_model_dir = Path(asr_model_dir) if asr_model_dir else QWEN3_ASR_MODEL_PATH
        self.aligner_model_dir = (
            Path(aligner_model_dir) if aligner_model_dir else QWEN3_ALIGNER_MODEL_PATH
        )
        self.language = language
        self.device = device
        self.low_memory = low_memory
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = vad_min_speech_ms
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.prompt = prompt
        self.need_word_time_stamp = need_word_time_stamp
        self.process: Optional[subprocess.Popen[str]] = None

    @property
    def runner_path(self) -> Path:
        return Path(__file__).with_name("qwen3_asr_runner.py")

    def _validate_components(self) -> None:
        missing = missing_components(
            runtime_dir=self.runtime_python.parent.parent,
            asr_model_dir=self.asr_model_dir,
            aligner_model_dir=self.aligner_model_dir,
        )
        if missing:
            raise EnvironmentError(
                "Missing Qwen3-ASR components: " + ", ".join(missing)
            )
        if not self.runner_path.is_file():
            raise EnvironmentError(f"Qwen3-ASR runner not found: {self.runner_path}")

    def _build_command(self, audio_path: Path, output_path: Path) -> list[str]:
        command = [
            str(self.runtime_python),
            str(self.runner_path),
            "--audio",
            str(audio_path),
            "--output",
            str(output_path),
            "--asr-model",
            str(self.asr_model_dir),
            "--aligner-model",
            str(self.aligner_model_dir),
            "--device",
            self.device,
            "--vad-threshold",
            str(self.vad_threshold),
            "--vad-min-speech-ms",
            str(self.vad_min_speech_ms),
            "--vad-min-silence-ms",
            str(self.vad_min_silence_ms),
            "--vad-speech-pad-ms",
            str(self.vad_speech_pad_ms),
        ]
        if self.language:
            command.extend(["--language", self.language])
        if self.prompt:
            command.extend(["--context", self.prompt])
        if self.low_memory:
            command.append("--low-memory")
        if self.vad_filter:
            command.append("--vad-filter")
        return command

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> dict:
        callback = callback or (lambda _progress, _message: None)
        self._validate_components()

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            audio_path = temp_dir / "audio.wav"
            output_path = temp_dir / "result.json"
            if isinstance(self.audio_input, str):
                shutil.copy2(self.audio_input, audio_path)
            elif self.file_binary:
                audio_path.write_bytes(self.file_binary)
            else:
                raise ValueError("No audio data available")

            command = self._build_command(audio_path, output_path)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            logger.debug("Qwen3-ASR command: %s", command)

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                # The runner emits model/runtime diagnostics on stderr while
                # progress events are sent on stdout.  Keep both streams on
                # one pipe so a full stderr pipe cannot pause inference.
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            error_message = ""
            diagnostics: list[str] = []
            assert self.process.stdout is not None
            process = self.process
            try:
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        diagnostics.append(line)
                        # Keep a bounded diagnostic tail in case a runtime
                        # exits without emitting its structured error event.
                        del diagnostics[:-20]
                        logger.debug("Qwen3-ASR runner: %s", line)
                        continue
                    if event.get("type") == "progress":
                        callback(
                            int(event.get("progress", 0)),
                            str(event.get("message", "")),
                        )
                    elif event.get("type") == "error":
                        error_message = str(event.get("message", "Qwen3-ASR failed"))

                return_code = process.wait()
                if return_code != 0:
                    diagnostic_text = "\n".join(diagnostics)
                    raise RuntimeError(
                        error_message
                        or diagnostic_text
                        or f"Qwen3-ASR exited with {return_code}"
                    )
                if not output_path.is_file():
                    raise RuntimeError("Qwen3-ASR did not produce a result file")
                callback(100, "Qwen3-ASR completed")
                return json.loads(output_path.read_text(encoding="utf-8"))
            finally:
                # Ensure a cancelled/failed transcription cannot leave the
                # model process alive and holding GPU memory.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.process = None

    def _make_segments(self, resp_data: dict) -> list[ASRDataSeg]:
        word_segments = [
            ASRDataSeg(
                str(item.get("text", "")).strip(),
                int(item.get("start_time", 0)),
                int(item.get("end_time", 0)),
            )
            for item in resp_data.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        if self.need_word_time_stamp:
            return word_segments
        return group_word_segments(word_segments)

    def _get_key(self) -> str:
        settings = {
            "model": str(self.asr_model_dir),
            "aligner": str(self.aligner_model_dir),
            "language": self.language,
            "device": self.device,
            "low_memory": self.low_memory,
            "vad_filter": self.vad_filter,
            "vad_threshold": self.vad_threshold,
            "vad_min_speech_ms": self.vad_min_speech_ms,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "vad_speech_pad_ms": self.vad_speech_pad_ms,
            "prompt": self.prompt,
            "word_timestamps": self.need_word_time_stamp,
        }
        digest = hashlib.md5(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{self.crc32_hex}-{digest}"
