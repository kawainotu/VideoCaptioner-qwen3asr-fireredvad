"""Nonvisual checks of independent settings, VAD routing and real local inference."""

import json
import sys
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from videocaptioner.core.asr.mimo_asr import MiMoASR
from videocaptioner.core.asr.mimo_vad import check_vad_environment, detect_speech_regions
from videocaptioner.core.asr.mimo_vad_runner import FIRERED_DEFAULTS, detect_regions
from videocaptioner.core.asr.qwen3_vad_models import get_qwen3_vad_model
from videocaptioner.core.asr.transcribe import _create_mimo_asr
from videocaptioner.core.entities import TranscribeConfig, TranscribeModelEnum
from videocaptioner.ui.common.config import Config, cfg
from videocaptioner.ui.components.MiMoVADSettingWidget import mimo_vad_components
from videocaptioner.ui.task_factory import TaskFactory

PARAMS = {
    "vad_threshold": 0.65,
    "vad_min_speech_ms": 350,
    "vad_min_silence_ms": 100,
    "vad_speech_pad_ms": 170,
    "firered_vad_smooth_window_size": 7,
    "firered_vad_speech_threshold": 0.6,
    "firered_vad_min_speech_frame": 16,
    "firered_vad_max_speech_frame": 500,
    "firered_vad_min_silence_frame": 45,
    "firered_vad_merge_silence_frame": 12,
    "firered_vad_extend_speech_frame": 25,
    "firered_vad_chunk_max_frame": 10000,
}


def make_asr(**kwargs):
    from videocaptioner.core.llm.check_mimo import generate_tiny_probe_wav

    return MiMoASR(
        generate_tiny_probe_wav(), "test-key", "https://gateway.test/custom/v1", **kwargs
    )


@pytest.mark.parametrize("model", ["silero", "firered"])
def test_complete_settings_route_from_factory_to_detector(monkeypatch, model):
    changed = ["transcribe_model", "mimo_vad_model"] + ["mimo_" + key for key in PARAMS]
    original = {name: getattr(cfg, name).value for name in changed}
    try:
        cfg.transcribe_model.value = TranscribeModelEnum.MIMO_ASR
        cfg.mimo_vad_model.value = model
        for key, value in PARAMS.items():
            getattr(cfg, "mimo_" + key).value = value
        config = TaskFactory.create_transcribe_task("C:/media/test.wav").transcribe_config
        asr = _create_mimo_asr("tests/fixtures/audio/zh.mp3", config)
        for key, value in PARAMS.items():
            assert getattr(config, "mimo_" + key) == value
        assert asr.vad_model == model
        with (
            patch(
                "videocaptioner.core.asr.mimo_asr.check_vad_environment",
                return_value=(True, "ready"),
            ),
            patch(
                "videocaptioner.core.asr.mimo_asr.detect_speech_regions", return_value=[(1, 3)]
            ) as detect,
        ):
            assert asr._prepare_chunks("fake.wav", 4) == [(1, 3)]
        kwargs = detect.call_args.kwargs
        assert kwargs["vad_model"] == model
        for key, value in PARAMS.items():
            assert kwargs[key.removeprefix("vad_") if key.startswith("vad_") else key] == value
    finally:
        for name, value in original.items():
            getattr(cfg, name).value = value


@pytest.mark.parametrize("key,value", PARAMS.items())
def test_cache_separates_each_vad_parameter(key, value):
    assert make_asr()._get_key() != make_asr(**{key: value})._get_key()


def test_cache_separates_model():
    assert make_asr()._get_key() != make_asr(vad_model="firered")._get_key()


def test_independent_persistence_roundtrip(tmp_path):
    from qfluentwidgets import qconfig

    names = ["mimo_vad_model"] + ["mimo_" + key for key in PARAMS]
    original = {name: getattr(cfg, name).value for name in names}
    local = Config()
    local.set(local.mimo_vad_model, "firered", save=False)
    for key, value in PARAMS.items():
        local.set(getattr(local, "mimo_" + key), value, save=False)
    serialized = local.toDict()
    assert serialized["MiMoASR"]["VadModel"] == "firered"
    assert serialized["QwenASR"]["VadModel"] == cfg.qwen_asr_vad_model.value
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(serialized), encoding="utf-8")
    loaded = Config()
    # QConfig.load switches a global singleton; restore it after this isolated load.
    old = qconfig._cfg
    try:
        qconfig.load(str(path), loaded)
        for key, value in PARAMS.items():
            assert getattr(loaded, "mimo_" + key).value == value
            assert getattr(loaded, "mimo_" + key).group == "MiMoASR"
            assert getattr(loaded, "qwen_asr_" + key).group == "QwenASR"
    finally:
        qconfig._cfg = old
        for name, value in original.items():
            getattr(cfg, name).value = value


@pytest.mark.parametrize(
    "model,regions", [("firered", [(0, 5), (5, 10)]), ("silero", [(1, 2), (2.2, 3)])]
)
def test_vad_segmentation_is_not_merged_away(model, regions):
    asr = make_asr(vad_model=model, vad_min_silence_ms=100)
    with (
        patch(
            "videocaptioner.core.asr.mimo_asr.check_vad_environment", return_value=(True, "ready")
        ),
        patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions", return_value=regions),
    ):
        assert asr._prepare_chunks("fake.wav", 12) == regions


def test_disabled_vad_never_checks_or_loads_model():
    with (
        patch("videocaptioner.core.asr.mimo_asr.check_vad_environment") as ready,
        patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions") as detect,
    ):
        assert make_asr(vad_filter=False, vad_model="firered")._prepare_chunks("fake.wav", 61) == [
            (0, 25),
            (25, 50),
            (50, 61),
        ]
        ready.assert_not_called()
        detect.assert_not_called()


def test_missing_firered_is_actionable_and_never_falls_back(tmp_path):
    with patch("videocaptioner.core.asr.mimo_vad.is_firered_vad_model_ready", return_value=False):
        ready, message = check_vad_environment(vad_model="firered")
        assert not ready and "FireRed" in message
        audio = tmp_path / "audio.wav"
        audio.touch()
        with pytest.raises(RuntimeError, match="FireRed"):
            detect_speech_regions(str(audio), vad_model="firered")


def test_manager_has_no_recognition_models():
    for key in ("silero", "firered"):
        components = mimo_vad_components(get_qwen3_vad_model(key))
        assert all(c["kind"] in ("runtime", "firered-vad") for c in components)


def test_runner_firered_dispatch_all_parameters(monkeypatch):
    class Audio:
        shape = (16000,)
        dtype = "float32"

        def __len__(self):
            return 16000

        def __mul__(self, value):
            return self

        def astype(self, dtype):
            self.dtype = dtype
            return self

    audio = Audio()
    np = SimpleNamespace(clip=lambda audio, lo, hi: audio, int16="int16")
    monkeypatch.setitem(sys.modules, "numpy", np)
    sf = SimpleNamespace(read=lambda path: (audio, 16000))
    config = MagicMock()
    model = MagicMock()
    model.detect.return_value = ({"timestamps": [(-1, 0.4), (0.7, 2)]}, [])
    vad = MagicMock()
    vad.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "fireredvad", SimpleNamespace(FireRedVad=vad, FireRedVadConfig=config)
    )
    values = {key: PARAMS["firered_vad_" + key] for key in FIRERED_DEFAULTS}
    regions = detect_regions(
        "fake.wav",
        vad_model="firered",
        model_dir="weights",
        **{"firered_vad_" + k: v for k, v in values.items()},
    )
    assert regions == [{"start": 0, "end": 0.4}, {"start": 0.7, "end": 1}]
    config.assert_called_once_with(use_gpu=False, **values)
    assert model.detect.call_args.args[0][0].dtype == np.int16
    assert model.detect.call_args.args[0][1] == 16000
    vad.from_pretrained.assert_called_once_with("weights", config.return_value)


def test_real_firered_fixture_if_installed(tmp_path):
    ready, message = check_vad_environment(vad_model="firered")
    if not ready:
        pytest.skip(message)
    regions = detect_speech_regions("tests/fixtures/audio/zh.mp3", vad_model="firered")
    assert regions and 0 <= regions[0][0] < regions[0][1] <= 3
    silence = tmp_path / "silence.wav"
    with wave.open(str(silence), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(bytes(32000))
    assert detect_speech_regions(str(silence), vad_model="firered") == []


def test_default_consistency_and_reset_scope(monkeypatch):
    from videocaptioner.core.mimo_vad_defaults import MIMO_FIRERED_VAD_DEFAULTS
    from videocaptioner.ui.components.MiMoVADSettingWidget import reset_mimo_firered_defaults

    asr = make_asr()
    config = TranscribeConfig(transcribe_model=TranscribeModelEnum.MIMO_ASR)
    expected = dict(
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    assert MIMO_FIRERED_VAD_DEFAULTS == FIRERED_DEFAULTS == expected
    for name, value in expected.items():
        assert getattr(config, "mimo_firered_vad_" + name) == value
        assert asr.vad_options["firered_vad_" + name] == value
        assert getattr(cfg, "mimo_firered_vad_" + name).defaultValue == value
    calls = []
    monkeypatch.setattr(cfg, "set", lambda item, value: calls.append((item, value)))
    reset_mimo_firered_defaults()
    assert len(calls) == 8
    assert all(
        item.group == "MiMoASR" and item.name.startswith("FireRedVad") for item, value in calls
    )
    assert {item.name: value for item, value in calls} == {
        getattr(cfg, "mimo_firered_vad_" + name).name: value for name, value in expected.items()
    }
