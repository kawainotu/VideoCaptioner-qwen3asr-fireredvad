"""Qianwen asynchronous file ASR with temporary OSS upload and native timestamps.

Protocol: https://platform.qianwenai.com/docs/developer-guides/speech/asr
Upload: https://platform.qianwenai.com/docs/api-reference/more/upload-file-get-temporary-url
"""

import hashlib
import json
import math
import mimetypes
import re
import subprocess
import threading
import time
import uuid
import wave
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

from videocaptioner.core.utils.cache import get_asr_cache, is_cache_enabled

from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR

DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/api/v1"
DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-filetrans"
SUPPORTED_LANGUAGE_HINTS = frozenset(
    "zh en ja ko vi th id ms tl hi ar fr de es pt ru it nl sv da fi no el pl cs hu ro bg hr sk".split()
)


def validate_language(value):
    value = (value or "").strip()
    if value and value != "auto" and value not in SUPPORTED_LANGUAGE_HINTS:
        raise ValueError("Qwen 云端 ASR 不支持此语言提示，请选择自动识别或支持的语言")
    return value


def validate_recognition_options(hotwords="", vocabulary_id="", context=""):
    """Normalize the local line editor into the filetrans vocabulary mapping."""
    if not all(isinstance(value, str) for value in (hotwords, vocabulary_id, context)):
        raise ValueError("Qwen 热词、词表 ID 和上下文必须是文本")
    vocabulary = {}
    for number, line in enumerate(hotwords.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        term, weight = line, 4
        if "|" in line:
            term, raw_weight = line.rsplit("|", 1)
            term, raw_weight = term.strip(), raw_weight.strip()
            if not raw_weight.isascii() or not raw_weight.isdigit():
                raise ValueError(f"Qwen 热词第 {number} 行的权重必须是整数 1–5 或 50")
            weight = int(raw_weight)
        if not term:
            raise ValueError(f"Qwen 热词第 {number} 行缺少词条")
        if weight not in (1, 2, 3, 4, 5, 50):
            raise ValueError(f"Qwen 热词第 {number} 行的权重必须是整数 1–5 或 50")
        if not term.isascii() and len(term) > 15:
            raise ValueError(f"Qwen 热词第 {number} 行含非 ASCII 字符，不能超过 15 个字符")
        if term.isascii() and len(term.split()) > 7:
            raise ValueError(f"Qwen 热词第 {number} 行不能超过 7 个英文单词")
        if term in vocabulary:
            raise ValueError(f"Qwen 热词第 {number} 行重复，请每个词条只填写一次")
        vocabulary[term] = weight
        if len(vocabulary) > 2000:
            raise ValueError("Qwen 热词最多填写 2000 条")
    if sum(weight == 50 for weight in vocabulary.values()) > 50:
        raise ValueError("Qwen 权重为 50 的超级热词最多填写 50 条")
    vocabulary_id = vocabulary_id.strip()
    if any(char.isspace() for char in vocabulary_id):
        raise ValueError("Qwen 词表 ID 不应包含空白字符")
    context = context.strip()
    if len(context) > 400:
        raise ValueError("Qwen 上下文最多填写 400 个字符，请精简后重试")
    return vocabulary, vocabulary_id, context


def normalize_base_url(value):
    value = (value or "").strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise ValueError("Qwen 云端 API Base URL 格式错误") from None
    if any(char.isspace() for char in value):
        raise ValueError("Qwen 云端 API Base URL 不允许包含空白")
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Qwen 云端 API Base URL 必须是无凭据、无查询参数的 HTTP(S) 基础地址")
    if parsed.path.endswith("/services/audio/asr/transcription") or parsed.path.endswith(
        "/compatible-mode/v1"
    ):
        raise ValueError("请填写 Qwen 文件转录基础地址（以 /api/v1 结尾）")
    return value


class _MultipartUpload:
    """A bounded-memory requests body; cancellation is checked between file chunks."""

    def __init__(self, path, fields, cancel):
        boundary = uuid.uuid4().hex
        self.content_type = f"multipart/form-data; boundary={boundary}"
        parts = []
        for key, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            )
        filename = "audio" + path.suffix.lower()
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
        )
        self.prefix = "".join(parts).encode()
        self.suffix = f"\r\n--{boundary}--\r\n".encode()
        self.path, self.cancel = path, cancel

    def __len__(self):
        return len(self.prefix) + self.path.stat().st_size + len(self.suffix)

    def __iter__(self):
        self.cancel()
        yield self.prefix
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                self.cancel()
                yield chunk
        self.cancel()
        yield self.suffix


class QwenFileTransASR(BaseASR):
    """One local audio file -> upload -> async task -> timestamped subtitles."""

    def __init__(
        self,
        audio_input,
        api_key,
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        language="",
        use_cache=False,
        need_word_time_stamp=False,
        task_timeout=1800,
        poll_interval=3,
        hotwords="",
        vocabulary_id="",
        context="",
    ):
        # BaseASR loads and decodes the entire audio; long file transcription must stream.
        self.audio_input = str(audio_input)
        self.path = Path(audio_input)
        self.api_key = (api_key or "").strip()
        self.base_url = normalize_base_url(base_url)
        self.model = (model or "").strip() or DEFAULT_MODEL
        self.language = validate_language(language)
        self.vocabulary, self.vocabulary_id, self.context = validate_recognition_options(
            hotwords, vocabulary_id, context
        )
        self.use_cache = use_cache
        self.need_word_time_stamp = need_word_time_stamp
        self.task_timeout = float(task_timeout)
        self.poll_interval = float(poll_interval)
        if (
            not math.isfinite(self.task_timeout)
            or self.task_timeout <= 0
            or not math.isfinite(self.poll_interval)
            or self.poll_interval <= 0
        ):
            raise ValueError("Qwen 转录超时和查询间隔必须为正数")
        self._cancelled = threading.Event()
        self._cache = get_asr_cache()
        self._deadline = None

    def cancel(self):
        self._cancelled.set()

    def _check_cancel(self):
        if self._cancelled.is_set():
            raise RuntimeError("Qwen 云端转录已取消（已提交的云端任务可能仍会继续）")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise RuntimeError("Qwen 云端转录等待超时，请稍后重试或增加 task_timeout")

    def _wait(self, seconds):
        self._check_cancel()
        if self._deadline is not None:
            seconds = min(seconds, max(0, self._deadline - time.monotonic()))
        self._cancelled.wait(seconds)
        self._check_cancel()

    def _safe_error(self, value):
        value = str(value).replace(self.api_key, "[已隐藏]") if self.api_key else str(value)
        for private in sorted(
            [self.context, self.vocabulary_id, *self.vocabulary], key=len, reverse=True
        ):
            if private:
                value = value.replace(private, "[已隐藏]")
        return re.sub(r"(?:https?|oss)://\S+", "[地址已隐藏]", value)[:500]

    def _request(self, session, method, url, stage, retry=False, retry_rate_limit=False, **kwargs):
        for attempt in range(3):
            self._check_cancel()
            remaining = max(0.1, self._deadline - time.monotonic()) if self._deadline else 30
            try:
                response = session.request(
                    method,
                    url,
                    timeout=(min(10, remaining), min(30, remaining)),
                    allow_redirects=False,
                    **kwargs,
                )
            except requests.RequestException:
                self._check_cancel()
                if retry and attempt < 2:
                    self._wait(2**attempt)
                    continue
                raise RuntimeError(f"Qwen {stage}网络连接失败；请检查网络和 API 地址") from None
            try:
                self._check_cancel()
            except RuntimeError:
                response.close()
                raise
            if 200 <= response.status_code < 300:
                return response
            # POST submission is retried only after an explicit rate-limit rejection.
            if (
                (retry or retry_rate_limit)
                and response.status_code == 429
                or retry
                and response.status_code in (500, 502, 503, 504)
            ) and attempt < 2:
                raw_delay = response.headers.get("Retry-After")
                delay = 2**attempt
                if raw_delay:
                    try:
                        delay = float(raw_delay)
                    except ValueError:
                        try:
                            delay = max(
                                0, parsedate_to_datetime(raw_delay).timestamp() - time.time()
                            )
                        except (ValueError, TypeError, OverflowError):
                            response.close()
                            raise RuntimeError(
                                "Qwen 返回了无效的 Retry-After，请稍后重试"
                            ) from None
                    if not math.isfinite(delay) or delay < 0:
                        response.close()
                        raise RuntimeError("Qwen 返回了无效的 Retry-After，请稍后重试")
                response.close()
                self._wait(delay)
                continue
            detail = ""
            try:
                error = response.json()
                if isinstance(error, dict):
                    detail = self._safe_error(
                        f"{error.get('code', '')} {error.get('message', '')}"
                    ).strip()
            except ValueError:
                pass
            status = response.status_code
            response.close()
            raise RuntimeError(
                f"Qwen {stage}失败（HTTP {status}）" + (f"：{detail}" if detail else "")
            )

    def _json_request(self, session, method, url, stage, **kwargs):
        response = self._request(session, method, url, stage, **kwargs)
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"Qwen {stage}返回了无效 JSON") from None
        finally:
            response.close()
        if not isinstance(data, dict):
            raise RuntimeError(f"Qwen {stage}返回格式错误")
        return data

    def _headers(self):
        if not self.api_key:
            raise ValueError("请填写 Qwen 云端 ASR API Key")
        return {"Authorization": f"Bearer {self.api_key}"}

    def get_upload_policy(self, session):
        # Also used by the connection probe without run(); bound header-driven waits.
        if self._deadline is None:
            self._deadline = time.monotonic() + 30
        result = self._json_request(
            session,
            "GET",
            self.base_url + "/uploads",
            "获取上传凭证",
            retry=True,
            headers=self._headers(),
            params={"action": "getPolicy", "model": self.model},
        )
        data = result.get("data")
        required = ("policy", "signature", "upload_dir", "upload_host", "oss_access_key_id")
        if not isinstance(data, dict) or any(not data.get(k) for k in required):
            raise RuntimeError("Qwen 上传凭证不完整，请确认渠道支持临时 OSS 文件上传")
        self._validate_remote_url(data["upload_host"])
        return data

    @staticmethod
    def _validate_remote_url(url):
        try:
            parsed = urlsplit(str(url))
            parsed.port
        except ValueError:
            raise RuntimeError("Qwen 返回了无效的 HTTPS 文件地址") from None
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise RuntimeError("Qwen 返回了无效的 HTTPS 文件地址")

    def _upload(self, session):
        policy = self.get_upload_policy(session)
        try:
            limit = float(policy["max_file_size_mb"]) * 1024 * 1024
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Qwen 上传凭证缺少有效的文件大小限制") from None
        if self.path.stat().st_size > limit:
            raise ValueError(
                f"音频超过 Qwen 临时上传限制（{limit / 1024 / 1024:g} MB），请缩短音频后重试"
            )
        key = policy["upload_dir"].rstrip("/") + "/" + uuid.uuid4().hex + self.path.suffix.lower()
        fields = {
            "OSSAccessKeyId": policy["oss_access_key_id"],
            "Signature": policy["signature"],
            "policy": policy["policy"],
            "key": key,
            "success_action_status": "200",
            "x-oss-object-acl": policy.get("x_oss_object_acl", "private"),
            "x-oss-forbid-overwrite": policy.get("x_oss_forbid_overwrite", "true"),
        }
        body = _MultipartUpload(self.path, fields, self._check_cancel)
        response = self._request(
            session,
            "POST",
            policy["upload_host"],
            "上传音频",
            data=body,
            headers={"Content-Type": body.content_type, "Content-Length": str(len(body))},
        )
        response.close()
        return "oss://" + key

    def _get_key(self):
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                self._check_cancel()
                digest.update(chunk)
        settings = json.dumps(
            [
                self.base_url,
                self.model,
                self.language,
                self.need_word_time_stamp,
                "native-ms-v1",
                self.vocabulary,
                self.vocabulary_id,
                self.context,
            ],
            sort_keys=True,
        )
        return digest.hexdigest() + ":" + hashlib.sha256(settings.encode()).hexdigest()

    def _preflight_audio(self):
        if self.path.stat().st_size > 2 * 1024**3:
            raise ValueError("Qwen 云端 ASR 的音频不能超过 2 GB")
        try:
            if self.path.suffix.lower() == ".wav":
                with wave.open(str(self.path), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
            else:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(self.path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode:
                    raise ValueError()
                duration = float(result.stdout.strip())
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError()
        except (OSError, ValueError, EOFError, wave.Error, subprocess.TimeoutExpired):
            raise ValueError("无法读取音频时长，请确认音频有效，且已安装 FFmpeg/ffprobe") from None
        if duration > 12 * 3600:
            raise ValueError("Qwen 云端 ASR 的单个音频不能超过 12 小时")
        self._check_cancel()

    def run(self, callback=None, **kwargs):
        self._check_cancel()
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise ValueError("Qwen 转录需要非空的本地音频文件")
        self._headers()
        self._deadline = time.monotonic() + self.task_timeout
        self._preflight_audio()
        cache_key = f"QwenFileTransASR:{self._get_key()}"
        # The server-side vocabulary may change while retaining the same ID.
        cache_allowed = self.use_cache and not self.vocabulary_id and is_cache_enabled()
        if cache_allowed:
            cached = self._cache.get(cache_key, default=None)
            if cached is not None:
                segments = self._make_segments(cached)
                self._check_cancel()
                return ASRData(segments)
        # A session has no global bearer header: credentials never go to OSS or result URLs.
        with requests.Session() as session:
            data = self._run(callback, session=session)
        segments = self._make_segments(data)
        self._check_cancel()
        if cache_allowed:
            self._cache.set(cache_key, data, expire=86400 * 2)
        if callback:
            callback(100, "Qwen 转录完成")
        return ASRData(segments)

    def _run(self, callback=None, session=None, **kwargs):
        notify = callback or (lambda *_: None)
        notify(5, "上传音频到 Qwen 临时存储")
        file_url = self._upload(session)
        headers = self._headers()
        headers.update({"X-DashScope-Async": "enable", "X-DashScope-OssResourceResolve": "enable"})
        parameters = {"channel_id": [0]}
        audio_input = {"file_urls": [file_url]}
        if self.vocabulary:
            parameters["vocabulary"] = self.vocabulary
        if self.vocabulary_id:
            parameters["vocabulary_id"] = self.vocabulary_id
        if self.context:
            audio_input["context"] = [
                {"role": "user", "content": [{"type": "input_text", "text": self.context}]}
            ]
        if self.language and self.language != "auto":
            parameters["language_hints"] = [self.language]
        submitted = self._json_request(
            session,
            "POST",
            self.base_url + "/services/audio/asr/transcription",
            "提交转录",
            retry_rate_limit=True,
            headers=headers,
            json={
                "model": self.model,
                "input": audio_input,
                "parameters": parameters,
            },
        )
        submitted_output = submitted.get("output")
        task_id = submitted_output.get("task_id") if isinstance(submitted_output, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("Qwen 未返回转录任务编号")
        notify(20, "Qwen 云端正在转录")
        while True:
            result = self._json_request(
                session,
                "GET",
                self.base_url + "/tasks/" + quote(task_id, safe=""),
                "查询转录任务",
                retry=True,
                headers=self._headers(),
            )
            output = result.get("output", {})
            if not isinstance(output, dict):
                raise RuntimeError("Qwen 返回了无法识别的任务结果")
            status = output.get("task_status")
            if status == "SUCCEEDED":
                break
            if status in ("FAILED", "CANCELED", "CANCELLED", "UNKNOWN"):
                raise RuntimeError(
                    "Qwen 转录任务失败："
                    + self._safe_error(f"{output.get('code', '')} {output.get('message', status)}")
                )
            if status not in ("PENDING", "RUNNING"):
                raise RuntimeError("Qwen 返回了无法识别的任务状态")
            self._wait(self.poll_interval)
        results = output.get("results")
        if (
            not isinstance(results, list)
            or len(results) != 1
            or not isinstance(results[0], dict)
            or results[0].get("subtask_status") != "SUCCEEDED"
        ):
            raise RuntimeError("Qwen 音频转录子任务失败或结果不完整")
        url = results[0].get("transcription_url", "")
        self._validate_remote_url(url)
        notify(90, "下载 Qwen 转录结果")
        data = self._json_request(session, "GET", url, "下载转录结果", retry=True)
        self._check_cancel()
        return data

    def _make_segments(self, resp_data):
        transcripts = resp_data.get("transcripts")
        if not isinstance(transcripts, list):
            raise RuntimeError("Qwen 结果缺少 transcripts")
        top_text = resp_data.get("text", "")
        if not isinstance(top_text, str):
            raise RuntimeError("Qwen 转录文本格式错误")
        if not transcripts and top_text.strip():
            raise RuntimeError("Qwen 结果有文本但缺少句子时间戳")
        segments = []
        for transcript in transcripts:
            if not isinstance(transcript, dict):
                raise RuntimeError("Qwen transcripts 格式错误")
            sentences = transcript.get("sentences")
            if not isinstance(sentences, list):
                raise RuntimeError("Qwen 结果缺少句子时间戳")
            transcript_text = transcript.get("text", "")
            if not isinstance(transcript_text, str):
                raise RuntimeError("Qwen 转录文本格式错误")
            if not sentences and transcript_text.strip():
                raise RuntimeError("Qwen 结果有文本但缺少句子时间戳")
            for sentence in sentences:
                if not isinstance(sentence, dict):
                    raise RuntimeError("Qwen 句子格式错误")
                sentence_text = sentence.get("text")
                if not isinstance(sentence_text, str):
                    raise RuntimeError("Qwen 句子缺少有效的文本")
                items = sentence.get("words") if self.need_word_time_stamp else None
                # Some languages/results expose sentence timing only; preserve that native timing.
                if items is None or items == []:
                    items = [sentence]
                if not isinstance(items, list):
                    raise RuntimeError("Qwen 词时间戳格式错误")
                segment_count = len(segments)
                for item in items:
                    if not isinstance(item, dict):
                        raise RuntimeError("Qwen 时间戳条目格式错误")
                    text = item.get("text")
                    if not isinstance(text, str):
                        raise RuntimeError("Qwen 转录文本格式错误")
                    punctuation = item.get("punctuation", "")
                    text = text + (punctuation if isinstance(punctuation, str) else "")
                    if not text.strip():
                        continue
                    begin, end = item.get("begin_time"), item.get("end_time")
                    if (
                        isinstance(begin, bool)
                        or isinstance(end, bool)
                        or not isinstance(begin, (int, float))
                        or not isinstance(end, (int, float))
                        or not math.isfinite(begin)
                        or not math.isfinite(end)
                        or begin < 0
                        or end <= begin
                        or begin != int(begin)
                        or end != int(end)
                    ):
                        raise RuntimeError("Qwen 结果包含无效或缺失的时间戳")
                    segments.append(ASRDataSeg(text=text, start_time=int(begin), end_time=int(end)))
                if sentence_text.strip() and len(segments) == segment_count:
                    raise RuntimeError("Qwen 句子有文本但词级结果为空")
        return segments
