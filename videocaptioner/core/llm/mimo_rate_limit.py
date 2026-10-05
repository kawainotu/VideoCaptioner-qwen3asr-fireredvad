"""Process-local RPM pacing and server-directed 429 cooldowns.

Legacy TPM arguments are accepted for compatibility and have no scheduling effect.
"""

import hashlib
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("mimo_rate_limit")

DEFAULT_MIMO_RPM = 80
DEFAULT_MIMO_TPM = 8000
MAX_AUTOMATIC_WAIT = 600.0


@dataclass
class Reservation:
    timestamp: float


class MiMoRateLimiter:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.requests = deque()
        self.cooldown_until = 0.0
        self.next_request_at = 0.0

    def acquire(self, tokens, rpm, tpm, cancelled=lambda: False, on_wait=None):
        # tokens/tpm are obsolete, retained only for callers with saved settings.
        if rpm < 1:
            raise ValueError("MiMo RPM 必须大于零")
        last_notice = -math.inf
        while True:
            if cancelled():
                raise RuntimeError("MiMo ASR 任务已被用户取消")
            with self.lock:
                now = self.clock()
                while self.requests and self.requests[0].timestamp <= now - 60:
                    self.requests.popleft()
                delay = max(self.cooldown_until - now, self.next_request_at - now, 0.0)
                if len(self.requests) >= rpm:
                    delay = max(delay, self.requests[len(self.requests) - rpm].timestamp + 60 - now)
                if delay <= 0:
                    reservation = Reservation(now)
                    self.requests.append(reservation)
                    self.next_request_at = now + 60.0 / rpm
                    return reservation
                if delay > MAX_AUTOMATIC_WAIT:
                    raise RuntimeError(
                        "MiMo 服务要求较长限流等待，请稍后重试（本次任务未提前发送请求）。"
                    )
            if on_wait and now - last_notice >= 1:
                reason = "server_429" if self.cooldown_until > now else "local_rpm"
                on_wait(delay, reason)
                last_notice = now
            self.sleep(min(delay, 0.1))

    def correct_usage(self, reservation, response):
        """Compatibility no-op: response usage never affects RPM pacing."""
        return None

    def cooldown(self, seconds):
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, self.clock() + seconds)


_LIMITERS = {}
_LIMITERS_LOCK = threading.Lock()


def get_mimo_limiter(endpoint, model):
    # The provider aggregates keys on an account. Conservatively share all keys
    # for the same endpoint/model; the application cannot identify account IDs.
    parts = urlsplit(endpoint)
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and (parts.scheme.lower(), port) not in (("https", 443), ("http", 80)):
        host += f":{port}"
    canonical = parts.scheme.lower() + "://" + host + parts.path.rstrip("/")
    identity = hashlib.sha256((canonical + "\0" + model).encode()).hexdigest()
    with _LIMITERS_LOCK:
        return _LIMITERS.setdefault(identity, MiMoRateLimiter())


def retry_after_seconds(headers, wall_clock=time.time):
    raw = (headers.get("retry-after") or headers.get("Retry-After")) if headers else None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        try:
            target = parsedate_to_datetime(raw).timestamp()
            server_date = headers.get("date") or headers.get("Date")
            reference = (
                parsedate_to_datetime(server_date).timestamp() if server_date else wall_clock()
            )
            value = target - reference
        except (ValueError, TypeError, OverflowError):
            raise ValueError("MiMo Retry-After 无法解析，请稍后重试。") from None
    if not math.isfinite(value):
        raise ValueError("MiMo Retry-After 无效，请稍后重试。")
    return max(0.0, value)


def send_with_mimo_limits(
    send,
    limiter,
    audio_seconds,
    rpm=DEFAULT_MIMO_RPM,
    tpm=DEFAULT_MIMO_TPM,
    cancelled=lambda: False,
    on_wait=None,
    max_retries=4,
    chunk_index=None,
):
    # audio_seconds/tpm have no effect: this path deliberately controls RPM only.
    for attempt in range(max_retries + 1):
        limiter.acquire(0, rpm, tpm, cancelled, on_wait)
        if cancelled():
            raise RuntimeError("MiMo ASR 任务已被用户取消")
        try:
            response = send()
        except Exception as error:
            if getattr(error, "status_code", None) != 429:
                raise
            headers = getattr(getattr(error, "response", None), "headers", None)
            try:
                delay = retry_after_seconds(headers)
            except ValueError:
                logger.warning(
                    "MiMo HTTP 429 chunk=%s attempt=%d reason=server_429 wait_seconds=%.3f invalid_retry_after=true",
                    chunk_index if chunk_index is not None else "probe",
                    attempt + 1,
                    MAX_AUTOMATIC_WAIT,
                )
                limiter.cooldown(MAX_AUTOMATIC_WAIT)
                raise
            if delay is None:
                delay = min(30 * (2**attempt), 120) + random.uniform(0, 1)
            logger.warning(
                "MiMo HTTP 429 chunk=%s attempt=%d reason=server_429 wait_seconds=%.3f",
                chunk_index if chunk_index is not None else "probe",
                attempt + 1,
                delay,
            )
            limiter.cooldown(delay)
            if on_wait:
                on_wait(max(delay, 0.1), "server_429")
            if attempt == max_retries:
                raise
            if delay > MAX_AUTOMATIC_WAIT:
                raise RuntimeError(
                    "MiMo 服务要求较长限流等待，请稍后重试（本次任务未提前发送请求）。"
                ) from None
            continue
        if cancelled():
            raise RuntimeError("MiMo ASR 任务已被用户取消")
        return response
