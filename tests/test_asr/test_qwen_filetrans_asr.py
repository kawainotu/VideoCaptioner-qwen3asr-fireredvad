import json
import time
import wave
from email.parser import BytesParser
from email.utils import formatdate
from unittest.mock import Mock

import pytest
import requests

from videocaptioner.core.asr.qwen_filetrans_asr import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    QwenFileTransASR,
    _MultipartUpload,
    normalize_base_url,
    validate_recognition_options,
)


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "音频.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)
    return path


def response(data=None, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode()
    result._content_consumed = True
    return result


POLICY = {
    "data": {
        "policy": "policy",
        "signature": "signature",
        "upload_dir": "temporary/abc",
        "upload_host": "https://upload.example",
        "oss_access_key_id": "temporary-key",
        "x_oss_object_acl": "private",
        "x_oss_forbid_overwrite": "true",
        "max_file_size_mb": 100,
    }
}
TRANSCRIPT = {
    "transcripts": [
        {
            "channel_id": 0,
            "text": "Hello world.",
            "sentences": [
                {
                    "begin_time": 1200,
                    "end_time": 2900,
                    "text": "Hello world.",
                    "words": [
                        {"begin_time": 1200, "end_time": 1800, "text": "Hello", "punctuation": " "},
                        {"begin_time": 2100, "end_time": 2900, "text": "world", "punctuation": "."},
                    ],
                }
            ],
        }
    ]
}


class Session:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if isinstance(kwargs.get("data"), _MultipartUpload):
            prepared = requests.Request(
                method, url, headers=kwargs["headers"], data=kwargs["data"]
            ).prepare()
            assert prepared.headers["Content-Length"] == str(len(b"".join(prepared.body)))
            assert "Transfer-Encoding" not in prepared.headers
        item = next(self.replies)
        if isinstance(item, Exception):
            raise item
        return item


def happy_replies(transcript=TRANSCRIPT):
    return [
        response(POLICY),
        response(),
        response({"output": {"task_id": "id-1"}}),
        response({"output": {"task_status": "RUNNING"}}),
        response(
            {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [
                        {
                            "subtask_status": "SUCCEEDED",
                            "transcription_url": "https://result.example/a?signature=secret",
                        }
                    ],
                }
            }
        ),
        response(transcript),
    ]


def test_local_audio_upload_submit_poll_download_native_timing(audio_path, monkeypatch):
    session = Session(happy_replies())
    monkeypatch.setattr(requests, "Session", lambda: session)
    asr = QwenFileTransASR(audio_path, "secret-key", language="en")
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    progress = Mock()
    data = asr.run(progress)
    assert [(s.text, s.start_time, s.end_time) for s in data.segments] == [
        ("Hello world.", 1200, 2900)
    ]
    upload_policy, upload, submit, poll, *_ = session.calls
    assert upload_policy[2]["params"] == {"action": "getPolicy", "model": DEFAULT_MODEL}
    assert upload[2]["headers"].get("Authorization") is None
    assert submit[1] == DEFAULT_BASE_URL + "/services/audio/asr/transcription"
    assert submit[2]["headers"]["X-DashScope-Async"] == "enable"
    assert submit[2]["headers"]["X-DashScope-OssResourceResolve"] == "enable"
    assert submit[2]["json"]["model"] == DEFAULT_MODEL
    assert submit[2]["json"]["input"]["file_urls"][0].startswith("oss://temporary/abc/")
    assert submit[2]["json"]["parameters"] == {"channel_id": [0], "language_hints": ["en"]}
    assert poll[1] == DEFAULT_BASE_URL + "/tasks/id-1"
    assert "headers" not in session.calls[-1][2]
    progress.assert_called_with(100, "Qwen 转录完成")


def test_multipart_prepared_body_fields_and_file_are_exact(audio_path):
    body = _MultipartUpload(
        audio_path,
        {
            "OSSAccessKeyId": "temporary-key",
            "Signature": "sig",
            "key": "prefix/file.wav",
            "policy": "abc",
        },
        lambda: None,
    )
    prepared = requests.Request(
        "POST", "https://upload.example", headers={"Content-Type": body.content_type}, data=body
    ).prepare()
    wire = b"".join(prepared.body)
    message = BytesParser().parsebytes(
        ("Content-Type: " + body.content_type + "\r\n\r\n").encode() + wire
    )
    fields = {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.get_payload()
    }
    assert fields["file"] == audio_path.read_bytes()
    assert fields["Signature"] == b"sig"
    assert fields["OSSAccessKeyId"] == b"temporary-key"
    assert int(prepared.headers["Content-Length"]) == len(wire)
    assert b"".join(body) == wire  # fresh file handle for each iteration


def test_words_preserve_spaces_punctuation_and_original_silence(audio_path):
    asr = QwenFileTransASR(audio_path, "key", need_word_time_stamp=True)
    assert [(s.text, s.start_time, s.end_time) for s in asr._make_segments(TRANSCRIPT)] == [
        ("Hello ", 1200, 1800),
        ("world.", 2100, 2900),
    ]
    no_words = {
        "transcripts": [{"sentences": [{"text": "a", "begin_time": 5000, "end_time": 6000}]}]
    }
    assert asr._make_segments(no_words)[0].start_time == 5000


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"transcripts": None},
        {"transcripts": [None]},
        {"transcripts": [{"sentences": [None]}]},
        {"transcripts": [{"text": "lost", "sentences": []}]},
        {"text": "lost", "transcripts": []},
        {"transcripts": [{"sentences": [{"text": "a", "begin_time": 2000, "end_time": 1000}]}]},
        {
            "transcripts": [
                {"sentences": [{"text": "a", "begin_time": float("nan"), "end_time": 1000}]}
            ]
        },
        {"transcripts": [{"sentences": [{"text": "a"}]}]},
        {"transcripts": [{"sentences": [{"text": "a", "begin_time": 0.1, "end_time": 0.2}]}]},
        {"transcripts": [{"text": 1, "sentences": []}]},
    ],
)
def test_malformed_results_are_rejected(audio_path, data):
    with pytest.raises(RuntimeError):
        QwenFileTransASR(audio_path, "key")._make_segments(data)


def test_failed_subtask_is_rejected_even_if_task_succeeded(audio_path, monkeypatch):
    replies = happy_replies()
    replies[4] = response(
        {"output": {"task_status": "SUCCEEDED", "results": [{"subtask_status": "FAILED"}]}}
    )
    session = Session(replies)
    monkeypatch.setattr(requests, "Session", lambda: session)
    asr = QwenFileTransASR(audio_path, "key")
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    with pytest.raises(RuntimeError, match="子任务"):
        asr.run()


def test_invalid_result_does_not_cache_or_announce_success(audio_path, monkeypatch):
    session = Session(happy_replies({"transcripts": [{"text": "bad", "sentences": []}]}))
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr("videocaptioner.core.asr.qwen_filetrans_asr.is_cache_enabled", lambda: True)
    asr = QwenFileTransASR(audio_path, "key", use_cache=True)
    asr._cache = Mock()
    asr._cache.get.return_value = None
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    progress = Mock()
    with pytest.raises(RuntimeError):
        asr.run(progress)
    asr._cache.set.assert_not_called()
    assert not any(call.args[0] == 100 for call in progress.call_args_list)


def test_safe_errors_and_bounded_retries(audio_path, monkeypatch):
    asr = QwenFileTransASR(audio_path, "secret-key")
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    replies = [response({"message": "secret-key https://host/path?token=signed"}, 503)] * 3
    session = Session(replies)
    with pytest.raises(RuntimeError) as error:
        asr._request(session, "GET", "https://host", "查询", retry=True)
    assert len(session.calls) == 3
    assert "secret-key" not in str(error.value) and "signed" not in str(error.value)
    session = Session([requests.Timeout("secret-key https://host/?token=signed")])
    with pytest.raises(RuntimeError):
        asr._request(session, "POST", "https://host", "提交", retry_rate_limit=True)
    assert len(session.calls) == 1


def test_submit_retries_explicit_rate_limit_only_and_upload_does_not(audio_path, monkeypatch):
    asr = QwenFileTransASR(audio_path, "key")
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    session = Session([response({}, 429), response({})])
    asr._request(session, "POST", "https://host", "提交", retry_rate_limit=True)
    assert len(session.calls) == 2
    session = Session([response({}, 429)])
    with pytest.raises(RuntimeError):
        asr._request(session, "POST", "https://host", "上传")
    assert len(session.calls) == 1


def test_cancel_before_run_during_upload_and_poll_wait(audio_path):
    asr = QwenFileTransASR(audio_path, "key")
    asr.cancel()
    with pytest.raises(RuntimeError, match="取消"):
        asr.run()
    with pytest.raises(RuntimeError, match="取消"):
        list(_MultipartUpload(audio_path, {}, asr._check_cancel))
    asr = QwenFileTransASR(audio_path, "key")
    asr._deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="超时"):
        asr._wait(30)


def test_policy_size_check_happens_before_upload(audio_path):
    policy = json.loads(json.dumps(POLICY))
    policy["data"]["max_file_size_mb"] = 0.001
    session = Session([response(policy)])
    with pytest.raises(ValueError, match="上传限制"):
        QwenFileTransASR(audio_path, "key")._upload(session)
    assert len(session.calls) == 1


def test_duration_limit_rejects_before_network(audio_path, monkeypatch):
    audio = Mock()
    audio.__enter__ = Mock(return_value=audio)
    audio.__exit__ = Mock()
    audio.getnframes.return_value = 43201
    audio.getframerate.return_value = 1
    monkeypatch.setattr(wave, "open", lambda *_: audio)
    session_factory = Mock()
    monkeypatch.setattr(requests, "Session", session_factory)
    with pytest.raises(ValueError, match="12 小时"):
        QwenFileTransASR(audio_path, "key").run()
    session_factory.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://a:bad/api/v1",
        "https://user:secret@host/api/v1",
        "https://host/api/v1?q=secret",
        "https://host/compatible-mode/v1",
        "https://host /api/v1",
        "https://[bad",
    ],
)
def test_bad_endpoint_is_rejected_without_echoing_secrets(url):
    with pytest.raises(ValueError) as error:
        normalize_base_url(url)
    assert "secret" not in str(error.value)


def test_cache_fingerprint_varies_by_model_endpoint_language_and_word_mode(audio_path):
    baseline = QwenFileTransASR(audio_path, "one")._get_key()
    assert QwenFileTransASR(audio_path, "two")._get_key() == baseline
    for settings in (
        {"model": "other"},
        {"base_url": "https://other/api/v1"},
        {"language": "en"},
        {"need_word_time_stamp": True},
    ):
        assert QwenFileTransASR(audio_path, "one", **settings)._get_key() != baseline


def test_unsupported_language_is_rejected_before_network(audio_path):
    with pytest.raises(ValueError, match="语言提示"):
        QwenFileTransASR(audio_path, "key", language="yue")


@pytest.mark.parametrize("words", [[{}], [{"text": " ", "begin_time": 0, "end_time": 100}]])
def test_nonempty_sentence_cannot_be_silently_lost_in_word_mode(audio_path, words):
    data = {"transcripts": [{"sentences": [{"text": "a", "words": words}]}]}
    with pytest.raises(RuntimeError):
        QwenFileTransASR(audio_path, "key", need_word_time_stamp=True)._make_segments(data)


@pytest.mark.parametrize("header,expected", [("15", 15), (formatdate(1015, usegmt=True), 15)])
def test_retry_after_is_honored_for_seconds_and_http_date(
    audio_path, monkeypatch, header, expected
):
    monkeypatch.setattr(time, "time", lambda: 1000)
    asr = QwenFileTransASR(audio_path, "key")
    wait = Mock()
    monkeypatch.setattr(asr, "_wait", wait)
    rejected = response({}, 429)
    rejected.headers["Retry-After"] = header
    session = Session([rejected, response({})])
    asr._request(session, "POST", "https://host", "提交", retry_rate_limit=True)
    wait.assert_called_once_with(expected)


def test_retry_after_wait_cancellation_prevents_next_request(audio_path, monkeypatch):
    asr = QwenFileTransASR(audio_path, "key")
    rejected = response({}, 429)
    rejected.headers["Retry-After"] = "60"
    session = Session([rejected])
    original_wait = asr._wait

    def cancel_wait(seconds):
        asr.cancel()
        original_wait(seconds)

    monkeypatch.setattr(asr, "_wait", cancel_wait)
    with pytest.raises(RuntimeError, match="取消"):
        asr._request(session, "POST", "https://host", "提交", retry_rate_limit=True)
    assert len(session.calls) == 1


def test_connection_probe_oversized_retry_after_is_bounded(audio_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    asr = QwenFileTransASR(audio_path, "key")
    waits = []

    def fake_wait(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(asr._cancelled, "wait", fake_wait)
    rejected = response({}, 429)
    rejected.headers["Retry-After"] = "9999999"
    session = Session([rejected])
    with pytest.raises(RuntimeError, match="超时"):
        asr.get_upload_policy(session)
    assert waits == [30]
    assert len(session.calls) == 1


def test_hotwords_and_context_have_exact_filetrans_payload(audio_path, monkeypatch):
    session = Session(happy_replies())
    monkeypatch.setattr(requests, "Session", lambda: session)
    asr = QwenFileTransASR(
        audio_path,
        "key",
        hotwords="斯蒂格勒\n第三持存 | 5\nVideoCaptioner | 50",
        vocabulary_id="vocab-existing",
        context="  录音讨论斯蒂格勒的第三持存。  ",
    )
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    asr.run()
    submit = session.calls[2][2]["json"]
    assert submit["parameters"] == {
        "channel_id": [0],
        "vocabulary": {"斯蒂格勒": 4, "第三持存": 5, "VideoCaptioner": 50},
        "vocabulary_id": "vocab-existing",
    }
    assert submit["input"]["context"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "录音讨论斯蒂格勒的第三持存。"}],
        }
    ]
    assert "messages" not in submit["input"]
    assert "context" not in submit["parameters"]


def test_empty_enhancements_are_omitted(audio_path, monkeypatch):
    session = Session(happy_replies())
    monkeypatch.setattr(requests, "Session", lambda: session)
    asr = QwenFileTransASR(audio_path, "key", hotwords=" \n ", vocabulary_id=" ", context=" ")
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    asr.run()
    submit = session.calls[2][2]["json"]
    assert list(submit["input"]) == ["file_urls"]
    assert submit["parameters"] == {"channel_id": [0]}


@pytest.mark.parametrize(
    "options,phrase",
    [
        ({"hotwords": "词 | 0"}, "权重"),
        ({"hotwords": "词 | 6"}, "权重"),
        ({"hotwords": "词 | 4.5"}, "整数"),
        ({"hotwords": " | 4"}, "缺少词条"),
        ({"hotwords": "私密词条\n私密词条 | 5"}, "重复"),
        ({"hotwords": "甲" * 16}, "15"),
        ({"hotwords": "résumé" * 3}, "15"),
        ({"hotwords": "one two three four five six seven eight"}, "7"),
        ({"hotwords": "\n".join("term" + str(n) for n in range(2001))}, "2000"),
        ({"hotwords": "\n".join(f"term{n} | 50" for n in range(51))}, "50"),
        ({"context": "私" * 401}, "400"),
        ({"hotwords": []}, "文本"),
        ({"vocabulary_id": "vocab wrong"}, "空白"),
    ],
)
def test_invalid_enhancements_fail_before_any_network(audio_path, monkeypatch, options, phrase):
    session_factory = Mock()
    monkeypatch.setattr(requests, "Session", session_factory)
    with pytest.raises(ValueError, match=phrase) as error:
        QwenFileTransASR(audio_path, "key", **options).run()
    assert "私密词条" not in str(error.value)
    session_factory.assert_not_called()


def test_hotwords_context_limits_accept_the_documented_boundaries():
    vocabulary, vocabulary_id, context = validate_recognition_options(
        "\n".join(
            ["甲" * 15 + " | 5", "one two three four five six seven"]
            + [f"term{n} | 50" for n in range(50)]
            + [f"ordinary{n}" for n in range(1948)]
        ),
        "vocab-existing",
        "甲" * 400,
    )
    assert len(vocabulary) == 2000 and len(context) == 400
    assert vocabulary["甲" * 15] == 5
    assert vocabulary["one two three four five six seven"] == 4
    assert vocabulary_id == "vocab-existing"


def test_cache_identity_uses_semantic_options_without_plaintext(audio_path):
    baseline = QwenFileTransASR(audio_path, "key")._get_key()
    variants = [
        {"hotwords": "私密术语"},
        {"hotwords": "私密术语 | 5"},
        {"context": "私密会议上下文"},
        {"vocabulary_id": "vocab-private"},
    ]
    keys = {QwenFileTransASR(audio_path, "key", **options)._get_key() for options in variants}
    assert len(keys) == len(variants) and baseline not in keys
    assert (
        QwenFileTransASR(audio_path, "key", hotwords="a\nb | 5")._get_key()
        == QwenFileTransASR(audio_path, "key", hotwords=" b | 5\n a | 4 \n")._get_key()
    )
    assert all("私密" not in key and "vocab-private" not in key for key in keys)


def test_remote_vocabulary_id_bypasses_cache_reads_and_writes(audio_path, monkeypatch):
    session = Session(happy_replies())
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr("videocaptioner.core.asr.qwen_filetrans_asr.is_cache_enabled", lambda: True)
    asr = QwenFileTransASR(audio_path, "key", vocabulary_id="vocab-existing", use_cache=True)
    asr._cache = Mock()
    monkeypatch.setattr(asr, "_wait", lambda _: None)
    asr.run()
    asr._cache.get.assert_not_called()
    asr._cache.set.assert_not_called()


def test_error_sanitizer_hides_enhancement_content(audio_path):
    asr = QwenFileTransASR(
        audio_path,
        "secret-key",
        hotwords="私密词条",
        vocabulary_id="vocab-private",
        context="私密会议背景",
    )
    message = asr._safe_error("secret-key 私密词条 vocab-private 私密会议背景 https://secret.url")
    assert "secret-key" not in message and "私密" not in message
    assert "vocab-private" not in message and "secret.url" not in message
