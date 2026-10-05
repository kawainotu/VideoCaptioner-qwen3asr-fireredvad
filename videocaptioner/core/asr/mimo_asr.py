"""Xiaomi MiMo ASR implementation.

Integrates Xiaomi MiMo speech recognition API (chat.completions with input_audio)
with local VAD filtering, bounded utterance chunking, timeline preservation,
and progress reporting.
"""

import hashlib
import io
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from openai import OpenAI
from pydub import AudioSegment

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.asr.base import BaseASR
from videocaptioner.core.asr.mimo_alignment import (
    align_mimo_segments,
    check_mimo_alignment_environment,
)
from videocaptioner.core.asr.mimo_vad import (
    check_vad_environment,
    detect_speech_regions,
    merge_speech_regions_into_chunks,
)
from videocaptioner.core.llm.check_mimo import (
    _sanitize_error,
    parse_mimo_response,
    validate_and_normalize_mimo_base_url,
)
from videocaptioner.core.llm.mimo_rate_limit import (
    DEFAULT_MIMO_RPM,
    DEFAULT_MIMO_TPM,
    get_mimo_limiter,
    send_with_mimo_limits,
)
from videocaptioner.core.mimo_vad_defaults import MIMO_FIRERED_VAD_DEFAULTS
from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("mimo_asr")

MAX_BASE64_BYTES = 10 * 1024 * 1024  # 10 MB limit as documented


class MiMoASR(BaseASR):
    """Xiaomi MiMo ASR implementation using OpenAI-compatible chat.completions API.

    Features:
    - Bounded chunking (e.g. 25s) before API submission
    - Local Silero or FireRed VAD filtering (with toggle, default ON)
    - Maps chunk start/end times back to the original media timeline
    - Zero API calls for silent audio when VAD is enabled
    - Enforces Base64 size limit (<= 10MB)
    - Safe error reporting and progress notification
    """

    def __init__(
        self,
        audio_input: Union[str, bytes],
        api_key: str,
        base_url: str,
        model: str = "mimo-v2.5-asr",
        language: str = "auto",
        vad_filter: bool = True,
        max_chunk_duration: float = 25.0,
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
        runtime_python: Optional[str] = None,
        client: Optional[OpenAI] = None,
        aligner_model_dir: Optional[str] = None,
        aligner_runtime_python: Optional[str] = None,
        aligner_device: str = "auto",
        rpm: int = DEFAULT_MIMO_RPM,
        tpm: int = DEFAULT_MIMO_TPM,
        vad_model: str = "silero",
        firered_vad_model_dir: Optional[str] = None,
        vad_threshold: float = 0.5,
        vad_min_speech_ms: int = 250,
        vad_min_silence_ms: int = 500,
        vad_speech_pad_ms: int = 300,
        firered_vad_smooth_window_size: int = MIMO_FIRERED_VAD_DEFAULTS["smooth_window_size"],
        firered_vad_speech_threshold: float = MIMO_FIRERED_VAD_DEFAULTS["speech_threshold"],
        firered_vad_min_speech_frame: int = MIMO_FIRERED_VAD_DEFAULTS["min_speech_frame"],
        firered_vad_max_speech_frame: int = MIMO_FIRERED_VAD_DEFAULTS["max_speech_frame"],
        firered_vad_min_silence_frame: int = MIMO_FIRERED_VAD_DEFAULTS["min_silence_frame"],
        firered_vad_merge_silence_frame: int = MIMO_FIRERED_VAD_DEFAULTS["merge_silence_frame"],
        firered_vad_extend_speech_frame: int = MIMO_FIRERED_VAD_DEFAULTS["extend_speech_frame"],
        firered_vad_chunk_max_frame: int = MIMO_FIRERED_VAD_DEFAULTS["chunk_max_frame"],
    ):
        # MiMo API does not provide word-level timestamps; always enforce False
        super().__init__(audio_input, use_cache=use_cache, need_word_time_stamp=False)

        self.raw_base_url = (base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip() or "mimo-v2.5-asr"
        self.language = self._normalize_language(language)
        self.aligner_model_dir = aligner_model_dir
        self.aligner_runtime_python = aligner_runtime_python
        self.aligner_device = aligner_device
        self._alignment_callback = None
        self._alignment_process = None
        self.rpm = rpm
        self.tpm = tpm
        self.vad_filter = vad_filter
        if vad_model not in ("silero", "firered"):
            raise ValueError(f"Unsupported VAD model: {vad_model}")
        self.vad_model = vad_model
        self.firered_vad_model_dir = firered_vad_model_dir
        self.vad_options = {
            "threshold": vad_threshold,
            "min_speech_ms": vad_min_speech_ms,
            "min_silence_ms": vad_min_silence_ms,
            "speech_pad_ms": vad_speech_pad_ms,
            "firered_vad_smooth_window_size": firered_vad_smooth_window_size,
            "firered_vad_speech_threshold": firered_vad_speech_threshold,
            "firered_vad_min_speech_frame": firered_vad_min_speech_frame,
            "firered_vad_max_speech_frame": firered_vad_max_speech_frame,
            "firered_vad_min_silence_frame": firered_vad_min_silence_frame,
            "firered_vad_merge_silence_frame": firered_vad_merge_silence_frame,
            "firered_vad_extend_speech_frame": firered_vad_extend_speech_frame,
            "firered_vad_chunk_max_frame": firered_vad_chunk_max_frame,
        }
        self.max_chunk_duration = max(5.0, min(float(max_chunk_duration), 30.0))
        self.runtime_python = runtime_python
        self._is_canceled = False
        self._client = client

        # Validate URL if provided
        try:
            self.base_url = validate_and_normalize_mimo_base_url(self.raw_base_url)
        except ValueError as err:
            logger.warning("Invalid MiMo Base URL: %s", _sanitize_error(str(err), self.api_key))
            self.base_url = ""

        if not self.base_url or not self.api_key:
            logger.warning("MiMo API Base URL or API Key is empty.")

    def cancel(self) -> None:
        """Cancel the ongoing transcription task."""
        self._is_canceled = True
        process = self._alignment_process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
            except OSError:
                pass

    def _register_alignment_process(self, process):
        self._alignment_process = process
        if process is not None and self._is_canceled:
            self.cancel()

    @staticmethod
    def _normalize_language(language: str) -> str:
        """Normalize language to MiMo supported values: auto, zh, en."""
        lang = (language or "").strip().lower()
        if lang in ("zh", "chinese", "中文"):
            return "zh"
        if lang in ("en", "english", "英语"):
            return "en"
        return "auto"

    @staticmethod
    def encode_audio_to_base64(wav_bytes: bytes) -> str:
        """Encode audio bytes to base64 and validate against the 10 MB limit."""
        import base64

        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
        if len(b64_audio) > MAX_BASE64_BYTES:
            raise ValueError(f"音频块大小 ({len(b64_audio)} 字节) 超过 MiMo API 10 MB 限制")
        return b64_audio

    def _get_key(self) -> str:
        """Get cache key including model, language, VAD, chunk duration, and endpoint hash.

        Does NOT include secret API keys or credentials.
        """
        url_hash = (
            hashlib.sha256(self.base_url.encode("utf-8")).hexdigest()[:12]
            if self.base_url
            else "none"
        )
        return (
            f"{self.crc32_hex}-{self.model}-{self.language}-"
            f"vad_{self.vad_filter}-{self.vad_model}-{hashlib.sha256(json.dumps(self.vad_options, sort_keys=True).encode()).hexdigest()[:16]}-chunk_{self.max_chunk_duration}-ep_{url_hash}"
        )

    def run(self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any) -> ASRData:
        """Override run to check cancellation and validate credentials before checking cache."""
        if self._is_canceled:
            raise RuntimeError("MiMo ASR 任务已被用户取消")
        if not self.base_url:
            raise ValueError("请先在设置中配置有效的 MiMo API Base URL")
        if not self.api_key:
            raise ValueError("请先在设置中配置 MiMo API Key")

        ready, message = check_mimo_alignment_environment(
            self.aligner_model_dir, self.aligner_runtime_python, self.aligner_device
        )
        if not ready:
            raise RuntimeError(message)
        self._alignment_callback = callback
        try:
            result = super().run(callback=callback, **kwargs)
            if self._is_canceled:
                raise RuntimeError("MiMo ASR 任务已被用户取消")
            if callback:
                callback(100, "本地对齐完成，短字幕已生成")
            return result
        finally:
            self._alignment_callback = None

    def _get_openai_client(self) -> OpenAI:
        """Create or return OpenAI client instance."""
        if self._client is not None:
            return self._client
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=60,
            max_retries=0,
        )

    def _prepare_chunks(
        self, temp_wav_path: str, total_duration_seconds: float
    ) -> List[tuple[float, float]]:
        """Determine chunk intervals (start_sec, end_sec) on the original timeline."""
        if self.vad_filter:
            is_ready, vad_msg = check_vad_environment(
                Path(self.runtime_python) if self.runtime_python else None,
                vad_model=self.vad_model,
                model_dir=self.firered_vad_model_dir,
            )
            if not is_ready:
                raise RuntimeError(f"本地 VAD 环境不可用: {vad_msg}")

            logger.info("Running local %s VAD speech detection...", self.vad_model)
            speech_regions = detect_speech_regions(
                temp_wav_path,
                runtime_python=self.runtime_python,
                vad_model=self.vad_model,
                model_dir=self.firered_vad_model_dir,
                **self.vad_options,
            )
            if not speech_regions:
                logger.info("VAD detected no speech (pure silence).")
                return []
            chunks = merge_speech_regions_into_chunks(
                speech_regions, max_chunk_duration=self.max_chunk_duration, max_gap_seconds=None
            )
            return chunks

        # VAD disabled: sequential bounded chunks
        chunks: List[tuple[float, float]] = []
        cursor = 0.0
        while cursor < total_duration_seconds:
            chunk_end = min(cursor + self.max_chunk_duration, total_duration_seconds)
            if chunk_end > cursor:
                chunks.append((round(cursor, 3), round(chunk_end, 3)))
            cursor = chunk_end
        return chunks

    def _run(self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any) -> dict:
        """Execute MiMo ASR recognition."""
        if not self.base_url:
            raise ValueError("请先在设置中配置有效的 MiMo API Base URL")
        if not self.api_key:
            raise ValueError("请先在设置中配置 MiMo API Key")

        if callback:
            callback(2, "正在准备音频数据...")

        # Load audio into pydub AudioSegment
        if self.file_binary:
            full_audio = AudioSegment.from_file(io.BytesIO(self.file_binary))
        elif isinstance(self.audio_input, str):
            full_audio = AudioSegment.from_file(self.audio_input)
        else:
            raise ValueError("No valid audio input available")

        # Standardize audio to 16kHz mono for optimal ASR and VAD accuracy
        full_audio = full_audio.set_frame_rate(16000).set_channels(1)
        total_duration_sec = full_audio.duration_seconds

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_wav = Path(temp_dir) / "full_audio_16k.wav"
            full_audio.export(str(temp_wav), format="wav")

            if self._is_canceled:
                raise RuntimeError("MiMo ASR 任务已被用户取消")

            if callback:
                callback(5, "正在进行语音片段检测...")

            chunk_intervals = self._prepare_chunks(str(temp_wav), total_duration_sec)

            if self._is_canceled:
                raise RuntimeError("MiMo ASR 任务已被用户取消")

            if not chunk_intervals:
                if callback:
                    callback(100, "未检测到有效语音片段")
                return {"segments": []}

            client = self._get_openai_client()
            total_chunks = len(chunk_intervals)
            raw_segments: list[dict[str, Any]] = []

            try:
                for idx, (start_sec, end_sec) in enumerate(chunk_intervals):
                    if self._is_canceled:
                        logger.info("MiMo ASR task canceled before chunk %d", idx + 1)
                        raise RuntimeError("MiMo ASR 任务已被用户取消")

                    start_ms = int(round(start_sec * 1000))
                    end_ms = int(round(end_sec * 1000))

                    chunk_audio = full_audio[start_ms:end_ms]
                    chunk_buf = io.BytesIO()
                    chunk_audio.export(chunk_buf, format="wav")
                    wav_bytes = chunk_buf.getvalue()

                    b64_audio = self.encode_audio_to_base64(wav_bytes)

                    if callback:
                        pct = 10 + int(60 * (idx / total_chunks))
                        callback(pct, f"正在识别语音片段 ({idx + 1}/{total_chunks})...")

                    logger.debug(
                        "Sending chunk %d/%d (%.2fs - %.2fs) to MiMo API",
                        idx + 1,
                        total_chunks,
                        start_sec,
                        end_sec,
                    )

                    def send_request():
                        return client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_audio",
                                            "input_audio": {
                                                "data": f"data:audio/wav;base64,{b64_audio}"
                                            },
                                        }
                                    ],
                                }
                            ],
                            extra_body={"asr_options": {"language": self.language}},
                        )

                    try:
                        response = send_with_mimo_limits(
                            send_request,
                            get_mimo_limiter(self.base_url, self.model),
                            chunk_audio.duration_seconds,
                            rpm=self.rpm,
                            tpm=self.tpm,
                            cancelled=lambda: self._is_canceled,
                            chunk_index=idx + 1,
                            on_wait=(
                                lambda delay, reason: callback(
                                    10 + int(60 * (idx / total_chunks)),
                                    (
                                        f"服务端返回 429：片段 {idx + 1}/{total_chunks} 将自动重试，等待约 {math.ceil(delay)} 秒，可取消..."
                                        if reason == "server_429"
                                        else f"控制请求频率（本地 RPM）：片段 {idx + 1}/{total_chunks} 约 {delay:.2f} 秒后发送，可取消..."
                                    ),
                                )
                            )
                            if callback
                            else None,
                        )

                        content = parse_mimo_response(response)
                        if content:
                            raw_segments.append(
                                {
                                    "text": content,
                                    "start": start_ms,
                                    "end": end_ms,
                                }
                            )
                    except Exception as exc:
                        clean_msg = _sanitize_error(str(exc), self.api_key)
                        logger.error(
                            "MiMo recognition failed on chunk %d: %s",
                            idx + 1,
                            clean_msg,
                        )
                        raise RuntimeError(f"MiMo 语音识别失败: {clean_msg}") from None

                    if self._is_canceled:
                        logger.info("MiMo ASR task canceled after chunk %d", idx + 1)
                        raise RuntimeError("MiMo ASR 任务已被用户取消")

            finally:
                # Close client if created locally and not passed externally
                if self._client is None and client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

            if callback:
                callback(70, "MiMo 识别完成，正在本地强制对齐原文...")

            if self._is_canceled:
                raise RuntimeError("MiMo ASR 任务已被用户取消")

            return {"segments": raw_segments}

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        """Align both new and cached MiMo raw transcripts before subtitle export."""
        raw = [
            dict(item, text=(item.get("text") or "").strip())
            for item in resp_data.get("segments", [])
            if (item.get("text") or "").strip()
        ]
        return align_mimo_segments(
            self.file_binary or self.audio_input,
            raw,
            model_dir=self.aligner_model_dir,
            runtime_python=self.aligner_runtime_python,
            device=self.aligner_device,
            language=self.language,
            cancelled=lambda: self._is_canceled,
            on_progress=self._alignment_callback,
            on_process=self._register_alignment_process,
            allow_vad_fallback=self.vad_filter,
        )
