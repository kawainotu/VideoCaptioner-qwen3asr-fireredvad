"""Nonvisual checks of strict original-text alignment and subtitle grouping."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from videocaptioner.core.asr import mimo_alignment as alignment
from videocaptioner.core.asr.asr_data import ASRData
from videocaptioner.core.asr.mimo_asr import MiMoASR
from videocaptioner.core.llm.check_mimo import generate_tiny_probe_wav


def record(text, start=1000, token_ms=150):
    tokens = [
        {"text": c, "start": i * token_ms / 1000, "end": (i + 1) * token_ms / 1000}
        for i, c in enumerate(text)
        if c.isalnum()
    ]
    return {
        "text": text,
        "start": start,
        "end": start + max(1, len(text)) * token_ms,
        "tokens": tokens,
    }


def test_long_chinese_short_cues_preserve_all_text_and_true_timing():
    text = (
        "这是一个完整的中文字幕测试，我们希望每条字幕长度合适，并且保留所有的标点和原始内容。" * 4
    )
    item = record(text)
    result = alignment.group_aligned_segment(item)
    assert len(result) >= 6
    assert "".join(s.text for s in result) == text
    assert all(alignment._width(s.text) <= 30 for s in result)
    assert all(s.start_time >= item["start"] and s.end_time <= item["end"] for s in result)
    assert all(a.end_time <= b.start_time for a, b in zip(result, result[1:]))
    assert all(0 < s.end_time - s.start_time <= 6000 for s in result)


def test_english_word_boundaries_and_punctuation_preserved():
    words = "This is an English sentence with many words to verify subtitle grouping".split() * 4
    text = " ".join(words) + "."
    item = {
        "text": text,
        "start": 500,
        "end": 30000,
        "tokens": [
            {"text": word, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, word in enumerate(words)
        ],
    }
    result = alignment.group_aligned_segment(item)
    assert "".join(s.text for s in result) == text
    assert all(alignment._width(s.text) <= 30 for s in result)
    assert [word for s in result for word in s.text.strip().rstrip(".").split()] == words


def test_digits_fullwidth_and_ligature_map_original_forms():
    item = {
        "text": "价格：１３.６０元，office ﬃ！",
        "start": 0,
        "end": 3000,
        "tokens": [
            {"text": "价格", "start": 0, "end": 0.4},
            {"text": "1360", "start": 0.4, "end": 1},
            {"text": "元", "start": 1, "end": 1.2},
            {"text": "office", "start": 1.2, "end": 2},
            {"text": "f", "start": 2, "end": 2.1},
            {"text": "f", "start": 2.1, "end": 2.2},
            {"text": "i", "start": 2.2, "end": 2.3},
        ],
    }
    mapped = alignment.map_alignment(item)
    assert "".join(s.text for s in mapped) == item["text"]
    assert mapped[-1].text == "ﬃ！" and mapped[-1].start_time == 2000


def test_zero_and_overlapping_tokens_merge_without_fake_word_times():
    item = {
        "text": "你好，世界！",
        "start": 1000,
        "end": 3000,
        "tokens": [
            {"text": "你", "start": 0, "end": 0},
            {"text": "好", "start": 0.1, "end": 0.4},
            {"text": "世", "start": 0.3, "end": 0.6},
            {"text": "界", "start": 0.6, "end": 0.6},
        ],
    }
    result = alignment.group_aligned_segment(item)
    assert [(s.text, s.start_time, s.end_time) for s in result] == [("你好，世界！", 1100, 1600)]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x["tokens"].pop(),
        lambda x: x["tokens"][0].update(start=float("nan")),
        lambda x: x.update(end=x["start"]),
        lambda x: [t.update(start=0, end=0) for t in x["tokens"]],
    ],
)
def test_untrustworthy_alignment_fails_explicitly(mutate):
    item = record("完整原文")
    mutate(item)
    with pytest.raises(ValueError):
        alignment.group_aligned_segment(item)


def test_single_atom_over_six_seconds_rejected():
    item = {
        "text": "hello",
        "start": 0,
        "end": 9000,
        "tokens": [{"text": "hello", "start": 0, "end": 9}],
    }
    with pytest.raises(ValueError):
        alignment.group_aligned_segment(item)


def test_cross_vad_gap_and_srt_multiple_entries():
    first = alignment.group_aligned_segment(
        record("中文长字幕需要拆成连续短字幕保留全部内容" * 3, start=1000)
    )
    second = alignment.group_aligned_segment(record("第二段保留静音间隙", start=20000))
    data = ASRData(first + second)
    assert first[-1].end_time < second[0].start_time
    assert data.to_srt().count(" --> ") == len(first) + len(second) > 2


def test_cache_raw_hit_still_aligns_without_api(monkeypatch):
    from videocaptioner.core.asr import base, mimo_asr

    monkeypatch.setattr(mimo_asr, "check_mimo_alignment_environment", lambda *a: (True, "ready"))
    item = record("缓存原文" * 15, start=0, token_ms=1)
    grouped = alignment.group_aligned_segment(item)
    align = MagicMock(return_value=grouped)
    monkeypatch.setattr(mimo_asr, "align_mimo_segments", align)
    monkeypatch.setattr(base, "is_cache_enabled", lambda: True)
    asr = MiMoASR(generate_tiny_probe_wav(), "fake-key", "https://service.test/v1", use_cache=True)
    cache = MagicMock()
    cache.get.return_value = {"segments": [{"text": item["text"], "start": 0, "end": 60}]}
    monkeypatch.setattr(asr, "_cache", cache)
    api = MagicMock()
    monkeypatch.setattr(asr, "_run", api)
    assert asr.run().segments == grouped
    api.assert_not_called()
    align.assert_called_once()


def test_missing_aligner_preflight_before_api(monkeypatch):
    from videocaptioner.core.asr import mimo_asr

    monkeypatch.setattr(
        mimo_asr, "check_mimo_alignment_environment", lambda *a: (False, "missing aligner")
    )
    asr = MiMoASR(generate_tiny_probe_wav(), "fake-key", "https://service.test/v1")
    api = MagicMock()
    monkeypatch.setattr(asr, "_run", api)
    with pytest.raises(RuntimeError, match="missing aligner"):
        asr.run()
    api.assert_not_called()


def test_cancel_before_alignment_without_subprocess(monkeypatch):
    monkeypatch.setattr(alignment, "check_mimo_alignment_environment", lambda *a: (True, "ready"))
    process = MagicMock()
    monkeypatch.setattr(alignment.subprocess, "Popen", process)
    with pytest.raises(RuntimeError, match="取消"):
        alignment.align_mimo_segments(
            generate_tiny_probe_wav(),
            [{"text": "你", "start": 0, "end": 10}],
            cancelled=lambda: True,
        )
    process.assert_not_called()


def test_real_local_aligner_fixture_if_ready():
    ready, message = alignment.check_mimo_alignment_environment()
    if not ready:
        pytest.skip(message)
    text = "今天深圳天氣怎麼樣?"
    result = alignment.align_mimo_segments(
        "tests/fixtures/audio/zh.mp3",
        [{"text": text, "start": 0, "end": 2628}],
        device="cpu",
        language="zh",
    )
    assert "".join(s.text for s in result) == text
    assert all(0 <= s.start_time < s.end_time <= 2628 for s in result)


def test_cancel_runner_cleans_process_and_temporary_files(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(alignment, "check_mimo_alignment_environment", lambda *a: (True, "ready"))
    process = MagicMock()
    process.poll.return_value = None
    paths = []

    def spawn(command, **kwargs):
        paths.append(Path(command[command.index("--manifest") + 1]).parent)
        return process

    monkeypatch.setattr(
        alignment.AudioSegment,
        "from_file",
        lambda *a, **k: alignment.AudioSegment.silent(duration=100, frame_rate=16000),
    )
    monkeypatch.setattr(alignment.subprocess, "Popen", spawn)
    checks = [False, True]
    registration = MagicMock()
    with pytest.raises(RuntimeError, match="取消"):
        alignment.align_mimo_segments(
            generate_tiny_probe_wav(),
            [{"text": "你", "start": 0, "end": 10}],
            cancelled=lambda: checks.pop(0),
            on_process=registration,
        )
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    assert registration.call_args_list[0].args == (process,)
    assert registration.call_args_list[-1].args == (None,)
    assert paths and not paths[0].exists()


def test_mimo_cancel_terminates_registered_aligner():
    asr = MiMoASR(generate_tiny_probe_wav(), "fake-key", "https://service.test/v1")
    process = MagicMock()
    process.poll.return_value = None
    asr._register_alignment_process(process)
    asr.cancel()
    process.terminate.assert_called_once()
    assert asr._is_canceled


def test_alignment_failure_preserves_only_raw_recognition_cache(monkeypatch):
    from videocaptioner.core.asr import base, mimo_asr

    monkeypatch.setattr(mimo_asr, "check_mimo_alignment_environment", lambda *a: (True, "ready"))
    monkeypatch.setattr(
        mimo_asr, "align_mimo_segments", MagicMock(side_effect=ValueError("mapping failed"))
    )
    monkeypatch.setattr(base, "is_cache_enabled", lambda: True)
    asr = MiMoASR(generate_tiny_probe_wav(), "fake-key", "https://service.test/v1", use_cache=True)
    raw = {"segments": [{"text": "你", "start": 0, "end": 10}]}
    monkeypatch.setattr(asr, "_run", lambda callback=None, **kwargs: raw)
    cache = MagicMock()
    cache.get.return_value = None
    monkeypatch.setattr(asr, "_cache", cache)
    with pytest.raises(ValueError, match="mapping failed"):
        asr.run()
    assert cache.set.call_args.args[1] == raw


def test_runner_loads_only_aligner_once_for_all_segments(monkeypatch):
    import sys

    from videocaptioner.core.asr.mimo_alignment_runner import align_manifest

    class Audio:
        ndim = 1

        def __getitem__(self, key):
            return self

        def __len__(self):
            return 100

    token = SimpleNamespace(text="你", start_time=0, end_time=0.1)
    model = MagicMock()
    model.align.return_value = [[token]]
    factory = MagicMock()
    factory.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=factory))
    monkeypatch.setitem(
        sys.modules, "soundfile", SimpleNamespace(read=lambda *a, **k: (Audio(), 16000))
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float32="float32"),
    )
    result = align_manifest(
        {
            "audio": "fake.wav",
            "language": "zh",
            "segments": [
                {"text": "你", "start": 0, "end": 100},
                {"text": "你", "start": 200, "end": 300},
            ],
        },
        "model",
        "cpu",
    )
    factory.from_pretrained.assert_called_once()
    assert model.align.call_count == 2 and len(result) == 2


def test_real_repeated_fixture_pause_and_short_cues_if_ready():
    import io

    from pydub import AudioSegment

    ready, message = alignment.check_mimo_alignment_environment()
    if not ready:
        pytest.skip(message)
    clip = (
        AudioSegment.from_file("tests/fixtures/audio/zh.mp3").set_frame_rate(16000).set_channels(1)
    )
    text = "今天深圳天氣怎麼樣?"
    sound = AudioSegment.empty()
    segments = []
    for _ in range(3):
        start = len(sound)
        sound += clip
        segments.append({"text": text, "start": start, "end": len(sound)})
        sound += AudioSegment.silent(duration=1000, frame_rate=16000)
    stream = io.BytesIO()
    sound.export(stream, format="wav")
    result = alignment.align_mimo_segments(stream.getvalue(), segments, device="cpu", language="zh")
    assert "".join(s.text for s in result) == text * 3
    assert len(result) == 3
    for original, cue in zip(segments, result):
        assert original["start"] <= cue.start_time < cue.end_time <= original["end"]
    assert all(a.end_time < b.start_time for a, b in zip(result, result[1:]))


@pytest.mark.parametrize("text,start,end", [("好。", 2261340, 2263220), ("嗯。", 4018050, 4018670)])
def test_authorized_short_reply_uses_original_vad_sentence_boundaries(
    monkeypatch, text, start, end
):
    import copy

    item = {
        "text": text,
        "start": start,
        "end": end,
        "tokens": [{"text": text[0], "start": 0, "end": 0}],
    }
    before = copy.deepcopy(item)
    logger = MagicMock()
    monkeypatch.setattr(alignment, "logger", logger)
    result = alignment.validate_aligned_segments([item], [item], allow_vad_fallback=True)
    assert [(s.text, s.start_time, s.end_time) for s in result] == [(text, start, end)]
    assert result[0].timing_source == "vad_sentence"
    assert result[0].needs_review is True
    assert result[0].source_segment_index == 1
    assert item == before  # Tokens remain all zero; no fabricated word spans.
    assert logger.warning.call_count == 2
    assert logger.warning.call_args_list[0].args[1:] == (1, start, end)
    assert "需复核" in logger.warning.call_args_list[0].args[0]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(
            text="这不是短应答", tokens=[{"text": "这不是短应答", "start": 0, "end": 0}]
        ),
        lambda x: x.update(end=x["start"] + 3001),
        lambda x: x.update(start=-1),
        lambda x: x.update(end=x["start"]),
        lambda x: x.update(tokens=[]),
        lambda x: x["tokens"][0].update(text="漏字"),
        lambda x: x["tokens"][0].update(start=float("nan"), end=float("nan")),
        lambda x: x["tokens"][0].update(start=float("inf"), end=float("inf")),
        lambda x: x["tokens"][0].update(start=-1, end=-1),
        lambda x: x["tokens"][0].update(start=4, end=4),
        lambda x: x["tokens"][0].update(start=0.4, end=0.1),
    ],
)
def test_vad_sentence_fallback_never_hides_invalid_or_incomplete_alignment(mutate):
    item = {
        "text": "嗯。",
        "start": 1000,
        "end": 2000,
        "tokens": [{"text": "嗯", "start": 0, "end": 0}],
    }
    mutate(item)
    with pytest.raises(ValueError):
        alignment.validate_aligned_segments([item], [item], allow_vad_fallback=True)


def test_vad_disabled_short_reply_still_fails_strictly():
    item = {
        "text": "嗯。",
        "start": 1000,
        "end": 2000,
        "tokens": [{"text": "嗯", "start": 0, "end": 0}],
    }
    with pytest.raises(ValueError, match="没有有效字词时间"):
        alignment.validate_aligned_segments([item], [item])


def test_vad_fallback_preserves_neighbors_and_silence_without_merging():
    first = record("前句。", start=1000)
    last = record("后句。", start=10000)
    short = {
        "text": "好。",
        "start": 5000,
        "end": 6880,
        "tokens": [{"text": "好", "start": 0, "end": 0}],
    }
    originals = [first, short, last]
    result = alignment.validate_aligned_segments(originals, originals, allow_vad_fallback=True)
    assert "".join(s.text for s in result) == "前句。好。后句。"
    assert result[0].end_time < result[1].start_time < result[1].end_time < result[2].start_time
    assert (result[1].start_time, result[1].end_time) == (5000, 6880)
    assert not hasattr(result[0], "needs_review")
    with pytest.raises(ValueError, match="缺少原始片段"):
        alignment.validate_aligned_segments(originals, [first, last], allow_vad_fallback=True)
    with pytest.raises(ValueError, match="不一致"):
        alignment.validate_aligned_segments(
            originals, [last, short, first], allow_vad_fallback=True
        )


@pytest.mark.parametrize("vad_filter", [True, False])
def test_mimo_vad_flag_controls_sentence_fallback(monkeypatch, vad_filter):
    from videocaptioner.core.asr import mimo_asr

    aligner = MagicMock(return_value=[])
    monkeypatch.setattr(mimo_asr, "align_mimo_segments", aligner)
    asr = MiMoASR(
        generate_tiny_probe_wav(), "fake-key", "https://service.test/v1", vad_filter=vad_filter
    )
    asr._make_segments({"segments": [{"text": "好。", "start": 0, "end": 100}]})
    assert aligner.call_args.kwargs["allow_vad_fallback"] is vad_filter


def test_partial_zero_alignment_keeps_existing_word_timing():
    item = {
        "text": "你好。",
        "start": 1000,
        "end": 2000,
        "tokens": [{"text": "你", "start": 0, "end": 0}, {"text": "好", "start": 0.1, "end": 0.4}],
    }
    result = alignment.validate_aligned_segments([item], [item], allow_vad_fallback=True)
    assert [(s.text, s.start_time, s.end_time) for s in result] == [("你好。", 1100, 1400)]
    assert not hasattr(result[0], "needs_review")
