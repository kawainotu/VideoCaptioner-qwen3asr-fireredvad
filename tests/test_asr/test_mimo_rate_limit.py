"""Rate-limit verification uses a virtual clock and never sends real API requests."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import RateLimitError

from videocaptioner.core.llm import mimo_rate_limit as rate
from videocaptioner.core.llm.mimo_rate_limit import MiMoRateLimiter, send_with_mimo_limits


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def sleep(self, delay):
        self.now += delay


@pytest.fixture(autouse=True)
def _mock_local_alignment_for_recognition_tests(monkeypatch):
    from videocaptioner.core.asr import mimo_asr
    from videocaptioner.core.asr.asr_data import ASRDataSeg
    monkeypatch.setattr(mimo_asr, "check_mimo_alignment_environment", lambda *args: (True, "ready"))
    monkeypatch.setattr(mimo_asr, "align_mimo_segments", lambda audio, segments, **kwargs: [
        ASRDataSeg(item["text"], item["start"], item["end"]) for item in segments])


@pytest.fixture
def budget(monkeypatch):
    clock = Clock()
    limiter = MiMoRateLimiter(clock.read, clock.sleep)
    monkeypatch.setattr(rate.random, "uniform", lambda a, b: 0)
    return clock, limiter


def limited_error(retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "https://service.test/v1")
    )
    return RateLimitError("limited", response=response, body={})


def response(usage=None):
    return SimpleNamespace(usage=usage)


def test_smooth_pacing_ignores_tokens_and_tpm(budget):
    clock, limiter = budget
    first = limiter.acquire(4500, 80, 8000)
    second = limiter.acquire(4500, 80, 8000)
    assert first.timestamp == 0
    assert second.timestamp == pytest.approx(0.75)
    assert clock.now == pytest.approx(0.75)


def test_rpm_window_and_pacing(budget):
    clock, limiter = budget
    limiter.acquire(1, 2, 100000)
    limiter.acquire(1, 2, 100000)
    third = limiter.acquire(1, 2, 100000)
    assert third.timestamp == pytest.approx(60)


def test_shared_budget_all_keys_canonical_endpoint(monkeypatch):
    monkeypatch.setattr(rate, "_LIMITERS", {})
    first = rate.get_mimo_limiter("HTTPS://SERVICE.TEST:443/Custom/v1/", "mimo")
    assert first is rate.get_mimo_limiter("https://service.test/Custom/v1", "mimo")
    assert first is not rate.get_mimo_limiter("https://service.test/custom/v1", "mimo")
    assert first is not rate.get_mimo_limiter("https://service.test/Custom/v1", "other")


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"total_tokens": 0},
        {"total_tokens": math.nan},
        MagicMock(),
        {"total_tokens": True},
    ],
)
def test_usage_never_changes_rpm_schedule(budget, usage):
    clock, limiter = budget
    reservation = limiter.acquire(2500, 80, 8000)
    limiter.correct_usage(reservation, response(usage))
    assert limiter.next_request_at == pytest.approx(0.75)


def test_huge_usage_is_ignored(budget):
    clock, limiter = budget
    reservation = limiter.acquire(2500, 80, 1)
    limiter.correct_usage(reservation, response({"total_tokens": 10**12}))
    assert limiter.acquire(10**15, 80, 0).timestamp == pytest.approx(0.75)


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Retry-After": "2.5"}, 2.5),
        ({"retry-after": "-1"}, 0),
        (
            {
                "Retry-After": "Wed, 21 Oct 2015 07:28:10 GMT",
                "Date": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
            10,
        ),
    ],
)
def test_retry_after_seconds_and_http_date(headers, expected):
    assert rate.retry_after_seconds(headers) == expected


@pytest.mark.parametrize("value", ["NaN", "inf", "invalid"])
def test_bad_retry_after_rejected(value):
    with pytest.raises(ValueError):
        rate.retry_after_seconds({"Retry-After": value})


def test_retry_charges_budget_and_preserves_shared_cooldown(budget):
    clock, limiter = budget
    calls = []

    def send():
        calls.append(clock.now)
        if len(calls) == 1:
            raise limited_error(40)
        return response()

    assert send_with_mimo_limits(send, limiter, 1, tpm=100000).usage is None
    assert calls == pytest.approx([0, 40])
    assert len(limiter.requests) == 2
    assert limiter.cooldown_until == 40


def test_retry_exhaustion_sets_final_shared_cooldown(budget):
    clock, limiter = budget
    calls = []

    def send():
        calls.append(clock.now)
        raise limited_error()

    with pytest.raises(RateLimitError):
        send_with_mimo_limits(send, limiter, 1, tpm=100000, max_retries=2)
    assert calls == pytest.approx([0, 30, 90])
    assert limiter.cooldown_until == pytest.approx(210)


def test_non429_never_retried(budget):
    clock, limiter = budget
    send = MagicMock(side_effect=RuntimeError("network failure"))
    with pytest.raises(RuntimeError, match="network failure"):
        send_with_mimo_limits(send, limiter, 1)
    send.assert_called_once()


def test_wait_cancelled_without_request(budget):
    clock, limiter = budget
    limiter.cooldown(120)
    send = MagicMock()
    with pytest.raises(RuntimeError, match="取消"):
        send_with_mimo_limits(send, limiter, 1, cancelled=lambda: clock.now >= 0.3)
    send.assert_not_called()
    assert clock.now < 0.5


def test_429_backoff_cancelled_no_duplicate_request(budget):
    clock, limiter = budget
    send = MagicMock(side_effect=limited_error(120))
    with pytest.raises(RuntimeError, match="取消"):
        send_with_mimo_limits(send, limiter, 1, cancelled=lambda: clock.now >= 0.3)
    send.assert_called_once()


@pytest.mark.parametrize("tpm", [1, 0, -1, None, ""])
def test_old_tpm_settings_never_block_requests(budget, tpm):
    clock, limiter = budget
    send = MagicMock(return_value=response({"total_tokens": 10**12}))
    send_with_mimo_limits(send, limiter, 20, tpm=tpm)
    send_with_mimo_limits(send, limiter, 20, tpm=tpm)
    assert send.call_count == 2
    assert clock.now == pytest.approx(0.75)


def test_long_retry_after_never_sends_early(budget):
    clock, limiter = budget
    send = MagicMock(side_effect=limited_error(3600))
    with pytest.raises(RuntimeError, match="稍后重试"):
        send_with_mimo_limits(send, limiter, 1)
    assert clock.now == 0
    send.assert_called_once()
    assert limiter.cooldown_until == 3600


def test_probe_shares_budget_and_reports_reachable_429(monkeypatch, budget):
    from videocaptioner.core.llm import check_mimo as probe

    clock, limiter = budget
    monkeypatch.setattr(probe, "get_mimo_limiter", lambda endpoint, model: limiter)
    client = MagicMock()
    client.chat.completions.create.side_effect = limited_error(60)
    success, message = probe.check_mimo_connection(
        "https://service.test/v1", "fake-key", client=client
    )
    assert not success and "429" in message and "可达" in message
    assert len(limiter.requests) == 1 and limiter.cooldown_until == 60
    client.chat.completions.create.assert_called_once()
    client.chat.completions.create.reset_mock()
    success, message = probe.check_mimo_connection(
        "https://service.test/v1", "another-key", client=client, cancelled=lambda: True
    )
    assert not success and "取消" in message
    client.chat.completions.create.assert_not_called()


def test_full_asr_retry_one_segment_original_timing(monkeypatch, budget):
    from videocaptioner.core.asr import mimo_asr
    from videocaptioner.core.llm.check_mimo import generate_tiny_probe_wav

    clock, limiter = budget
    monkeypatch.setattr(mimo_asr, "get_mimo_limiter", lambda endpoint, model: limiter)
    client = MagicMock()
    success = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="text", refusal=None)
            )
        ],
    )
    client.chat.completions.create.side_effect = [limited_error(35), success]
    asr = mimo_asr.MiMoASR(
        generate_tiny_probe_wav(),
        "fake-key",
        "https://service.test/v1",
        client=client,
        vad_filter=False,
    )
    monkeypatch.setattr(asr, "_prepare_chunks", lambda path, duration: [(0.02, 0.08)])
    data = asr._run()
    assert data == {"segments": [{"text": "text", "start": 20, "end": 80}]}
    assert client.chat.completions.create.call_count == 2


def test_rpm_tpm_configuration_route_and_cache_exclusion(monkeypatch):
    from videocaptioner.core.asr.transcribe import _create_mimo_asr
    from videocaptioner.core.entities import TranscribeModelEnum
    from videocaptioner.ui.common.config import cfg
    from videocaptioner.ui.task_factory import TaskFactory

    names = ["mimo_rpm", "mimo_tpm", "transcribe_model"]
    original = {name: getattr(cfg, name).value for name in names}
    try:
        cfg.mimo_rpm.value, cfg.mimo_tpm.value = 50, 5000
        cfg.transcribe_model.value = TranscribeModelEnum.MIMO_ASR
        config = TaskFactory.create_transcribe_task("tests/fixtures/audio/zh.mp3").transcribe_config
        asr = _create_mimo_asr("tests/fixtures/audio/zh.mp3", config)
        assert (asr.rpm, asr.tpm) == (50, 5000)
        key = asr._get_key()
        asr.rpm, asr.tpm = 80, 8000
        assert asr._get_key() == key
        assert cfg.mimo_rpm.group == cfg.mimo_tpm.group == "MiMoASR"
    finally:
        for name, value in original.items():
            getattr(cfg, name).value = value


def test_two_segments_retry_does_not_resend_successful_segment(monkeypatch, budget):
    from videocaptioner.core.asr import mimo_asr
    from videocaptioner.core.llm.check_mimo import generate_tiny_probe_wav

    clock, limiter = budget
    monkeypatch.setattr(mimo_asr, "get_mimo_limiter", lambda endpoint, model: limiter)

    def recognized(text):
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=text, refusal=None)
                )
            ],
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        recognized("first"),
        limited_error(45),
        recognized("second"),
    ]
    asr = mimo_asr.MiMoASR(
        generate_tiny_probe_wav(),
        "fake-key",
        "https://service.test/v1",
        client=client,
        vad_filter=False,
    )
    monkeypatch.setattr(asr, "_prepare_chunks", lambda path, duration: [(0, 0.04), (0.05, 0.1)])
    assert asr._run() == {
        "segments": [
            {"text": "first", "start": 0, "end": 40},
            {"text": "second", "start": 50, "end": 100},
        ]
    }
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["messages"] != calls[1].kwargs["messages"]
    assert calls[1].kwargs == calls[2].kwargs


def test_retry_exhaustion_does_not_cache_partial_result(monkeypatch, budget):
    from videocaptioner.core.asr import base, mimo_asr
    from videocaptioner.core.llm.check_mimo import generate_tiny_probe_wav

    clock, limiter = budget
    monkeypatch.setattr(mimo_asr, "get_mimo_limiter", lambda endpoint, model: limiter)
    monkeypatch.setattr(base, "is_cache_enabled", lambda: True)
    success = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="first", refusal=None)
            )
        ],
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [success] + [limited_error(0)] * 5
    asr = mimo_asr.MiMoASR(
        generate_tiny_probe_wav(),
        "fake-key",
        "https://service.test/v1",
        client=client,
        vad_filter=False,
        use_cache=True,
    )
    monkeypatch.setattr(asr, "_prepare_chunks", lambda path, duration: [(0, 0.04), (0.05, 0.1)])
    cache = MagicMock()
    cache.get.return_value = None
    monkeypatch.setattr(asr, "_cache", cache)
    with pytest.raises(RuntimeError, match="limited"):
        asr.run()
    assert client.chat.completions.create.call_count == 6
    cache.set.assert_not_called()


@pytest.mark.parametrize("idle", [61, 12 * 3600])
def test_idle_first_request_has_no_rpm_wait(budget, idle):
    clock, limiter = budget
    limiter.acquire(9999, 80, 1)
    clock.sleep(idle)
    first = limiter.acquire(9999, 80, 1)
    assert first.timestamp == idle
    assert clock.now == idle


def test_request_duration_counts_toward_rpm_spacing(budget):
    clock, limiter = budget
    starts = []

    def send():
        starts.append(clock.now)
        clock.sleep(2)
        return response({"total_tokens": 10**12})

    send_with_mimo_limits(send, limiter, 20, tpm=1)
    send_with_mimo_limits(send, limiter, 20, tpm=1)
    assert starts == [0, 2]
    assert clock.now == 4


def test_rpm_wait_and_server_429_wait_have_distinct_reasons(budget):
    clock, limiter = budget
    notices = []
    limiter.acquire(0, 80, 0)
    limiter.acquire(0, 80, 0, on_wait=lambda delay, reason: notices.append((delay, reason)))
    assert notices[0] == (0.75, "local_rpm")
    notices.clear()
    send = MagicMock(side_effect=[limited_error(30), response()])
    send_with_mimo_limits(
        send, limiter, 20, on_wait=lambda delay, reason: notices.append((delay, reason))
    )
    assert any(reason == "server_429" and delay == 30 for delay, reason in notices)


def test_logs_only_actual_429_and_safe_metadata(monkeypatch, budget):
    clock, limiter = budget
    log = MagicMock()
    monkeypatch.setattr(rate.logger, "warning", log)
    send_with_mimo_limits(lambda: response(), limiter, 20, tpm=1, chunk_index=1)
    send_with_mimo_limits(lambda: response(), limiter, 20, tpm=1, chunk_index=2)
    log.assert_not_called()
    send = MagicMock(side_effect=[limited_error(30), response()])
    send_with_mimo_limits(send, limiter, 20, tpm=1, chunk_index=3)
    assert log.call_count == 1
    template, *args = log.call_args.args
    line = template % tuple(args)
    assert "HTTP 429" in line and "chunk=3" in line and "attempt=1" in line
    assert "reason=server_429" in line and "wait_seconds=30.000" in line
    assert "fake-key" not in line and "audio" not in line


def test_expired_shared_429_cooldown_after_idle_sends_immediately(budget):
    clock, limiter = budget
    limiter.acquire(0, 80, 0)
    limiter.cooldown(120)
    clock.sleep(12 * 3600)
    notices = []
    send = MagicMock(return_value=response({"total_tokens": 10**12}))
    send_with_mimo_limits(
        send, limiter, 20, tpm=1, on_wait=lambda delay, reason: notices.append((delay, reason))
    )
    send.assert_called_once()
    assert clock.now == 12 * 3600
    assert notices == []
