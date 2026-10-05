"""小米 MiMo-ASR 离线回归与集成测试套件。

覆盖：
1. Base URL 规范化、IPv6 支持与严格校验 (http/https, IPv6 [::1]:port, host, credentials/query/fragment/空白/端口越界)
2. 共享响应解析器 parse_mimo_response (正常文本、合法静音、null/missing content, refusal, length 截断)
3. 真实 OpenAI SDK 序列化与 httpx.MockTransport 请求体路径/参数验证
4. 探针测试 (空文本成功、异常 schema 报错、服务端返回文本不回显)
5. 异常脱敏清洗与无敏感凭据日志 (转录异常、探针异常、Worker 异常)
6. 缓存键验证 (包含 endpoint hash 与 chunk 限制，无 credentials) 与前置凭据校验
7. 显式取消机制多点检测 (前置取消、VAD后取消、分块后取消，无残缺结果，不写入缓存)
8. VAD 全静音 0 次 API 调用与时间轴保持 (1-3s, 10-12s 偏移不平移)
9. VAD 关闭时 61s 音频边界切块精确覆盖与 >25s 连续语音拆分
10. 单块 Base64 >10MB 大小限制检测
11. transcribe() 针对 MiMo 绕过 optimize_timing() 保留精确分块边界
12. MiMo 独立语言枚举隔离与 TaskFactory 映射 (不受全局日语等语言影响)
13. 配置持久化与序列化 Roundtrip (隔离测试，不写入用户真实配置)
14. 本地 Silero VAD 环境检测 (Windows 依赖检查) 与真实 Fixture 执行测试
15. Qt 线程生命周期与安全 Worker 测试 (使用 QCoreApplication ONLY，无任何视觉组件)
"""

import hashlib
import inspect
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import OpenAI
from PyQt5.QtCore import QCoreApplication

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.asr.mimo_asr import MiMoASR
from videocaptioner.core.asr.mimo_vad import (
    check_vad_environment,
    detect_speech_regions,
    merge_speech_regions_into_chunks,
)
from videocaptioner.core.asr.transcribe import transcribe
from videocaptioner.core.entities import (
    MiMoLanguageEnum,
    TranscribeConfig,
    TranscribeLanguageEnum,
    TranscribeModelEnum,
)
from videocaptioner.core.llm.check_mimo import (
    _sanitize_error,
    check_mimo_connection,
    generate_tiny_probe_wav,
    normalize_mimo_base_url,
    parse_mimo_response,
    validate_and_normalize_mimo_base_url,
)
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.MiMoASRSettingWidget import (
    _ACTIVE_MIMO_WORKERS,
    MiMoConnectionThread,
)
from videocaptioner.ui.task_factory import TaskFactory


@pytest.fixture(autouse=True)
def _mock_local_alignment_for_recognition_tests(monkeypatch):
    from videocaptioner.core.asr import mimo_asr
    from videocaptioner.core.asr.asr_data import ASRDataSeg
    monkeypatch.setattr(mimo_asr, "check_mimo_alignment_environment", lambda *args: (True, "ready"))
    monkeypatch.setattr(mimo_asr, "align_mimo_segments", lambda audio, segments, **kwargs: [
        ASRDataSeg(item["text"], item["start"], item["end"]) for item in segments])


@pytest.fixture(autouse=True)
def _isolated_mimo_rate_clock(monkeypatch):
    """Existing offline tests must not spend real time on new request pacing."""
    from videocaptioner.core.llm import mimo_rate_limit as rate
    clock = [0.0]
    cls = rate.MiMoRateLimiter
    monkeypatch.setattr(rate, "_LIMITERS", {})
    monkeypatch.setattr(rate, "MiMoRateLimiter", lambda: cls(
        clock=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay)))


# =====================================================================
# 辅助函数：生成可用于测试的有效 WAV 文件
# =====================================================================
def create_test_wav(file_path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    """生成包含简单 PCM 音频的 WAV 测试文件。"""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)


# =====================================================================
# 1. Base URL 规范化、IPv6 支持与严格校验
# =====================================================================
class TestMiMoUrlValidation:
    """测试 Base URL 的合法性校验与 IPv6/通道保留。"""

    def test_valid_urls_and_preserve_custom_channel(self):
        """测试合法的 URL 规范化，并去除尾随斜杠、保留自定义子路径。"""
        assert (
            validate_and_normalize_mimo_base_url("https://api.xiaomimimo.com/v1/")
            == "https://api.xiaomimimo.com/v1"
        )
        assert (
            validate_and_normalize_mimo_base_url("http://gateway.internal:8080/custom/token-plan/v2///")
            == "http://gateway.internal:8080/custom/token-plan/v2"
        )
        assert normalize_mimo_base_url("https://api.xiaomimimo.com/v1/") == "https://api.xiaomimimo.com/v1"

    def test_ipv6_url_preserves_brackets(self):
        """测试 IPv6 地址如 http://[::1]:8000/channel 正确保留方括号，不变成非法 http://::1:8000。"""
        ipv6_url = "http://[::1]:8000/channel/"
        normalized = validate_and_normalize_mimo_base_url(ipv6_url)
        assert normalized == "http://[::1]:8000/channel"
        assert "[::1]" in normalized
        assert "::1:" not in normalized.replace("[::1]:", "")

    def test_empty_returns_empty_string(self):
        """测试空值或纯空白安全返回空字符串。"""
        assert validate_and_normalize_mimo_base_url("") == ""
        assert validate_and_normalize_mimo_base_url("   ") == ""
        assert validate_and_normalize_mimo_base_url(None) == ""

    def test_whitespace_and_control_chars_rejected(self):
        """测试 URL 内部含有换行、制表符或空格等非法字符时拒绝。"""
        with pytest.raises(ValueError, match="包含非法空白或控制字符"):
            validate_and_normalize_mimo_base_url("https://api.xiaomimimo.com/v1\n/channel")
        with pytest.raises(ValueError, match="包含非法空白或控制字符"):
            validate_and_normalize_mimo_base_url("https://api. xiaomimimo.com/v1")
        with pytest.raises(ValueError, match="包含非法空白或控制字符"):
            validate_and_normalize_mimo_base_url("https://api.xiaomimimo.com/v1\x00")

    def test_invalid_scheme_raises(self):
        """测试非 http/https 协议报错。"""
        with pytest.raises(ValueError, match="Base URL 协议无效"):
            validate_and_normalize_mimo_base_url("ftp://api.xiaomimimo.com/v1")
        with pytest.raises(ValueError, match="Base URL 协议无效"):
            validate_and_normalize_mimo_base_url("api.xiaomimimo.com/v1")

    def test_missing_host_raises(self):
        """测试缺少有效主机名报错。"""
        with pytest.raises(ValueError, match="必须包含有效的主机名"):
            validate_and_normalize_mimo_base_url("https:///v1")

    def test_invalid_port_raises(self):
        """测试端口非法或超出 1-65535 报错。"""
        with pytest.raises(ValueError, match="端口号"):
            validate_and_normalize_mimo_base_url("http://127.0.0.1:99999/v1")
        with pytest.raises(ValueError, match="端口号"):
            validate_and_normalize_mimo_base_url("http://127.0.0.1:abc/v1")

    def test_credentials_in_url_rejected(self):
        """测试拒绝包含凭据的 URL。"""
        with pytest.raises(ValueError, match="不能包含用户名或密码"):
            validate_and_normalize_mimo_base_url("https://user:pass@api.xiaomimimo.com/v1")

    def test_query_or_fragment_rejected(self):
        """测试拒绝包含查询参数或片段标识符。"""
        with pytest.raises(ValueError, match="不能包含查询参数"):
            validate_and_normalize_mimo_base_url("https://api.xiaomimimo.com/v1?auth=token")
        with pytest.raises(ValueError, match="不能包含片段标识符"):
            validate_and_normalize_mimo_base_url("https://api.xiaomimimo.com/v1#heading")


# =====================================================================
# 2. 共享响应解析器严格校验
# =====================================================================
class TestMiMoResponseParser:
    """测试共享响应解析器 parse_mimo_response 对各类异常响应的拦截。"""

    def test_valid_text_and_valid_silence(self):
        """测试正常识别文本与合法静音。"""
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.refusal = None
        mock_choice.message.content = "   识别到的字幕内容   "
        mock_resp.choices = [mock_choice]
        assert parse_mimo_response(mock_resp) == "识别到的字幕内容"

        # 合法静音（空字符串）
        mock_choice.message.content = ""
        assert parse_mimo_response(mock_resp) == ""

    def test_missing_choices_or_empty_choices(self):
        """测试缺少 choices 或 choices 为空列表时明确报错。"""
        mock_resp = MagicMock(choices=[])
        with pytest.raises(ValueError, match="choices 为空"):
            parse_mimo_response(mock_resp)

        with pytest.raises(ValueError, match="choices 为空"):
            parse_mimo_response(None)

    def test_truncated_finish_reason_length(self):
        """测试由于长度限制被截断的响应报错。"""
        mock_resp = MagicMock()
        mock_choice = MagicMock(finish_reason="length")
        mock_resp.choices = [mock_choice]
        with pytest.raises(ValueError, match="finish_reason=length"):
            parse_mimo_response(mock_resp)

    def test_refusal_raises(self):
        """测试 API 明确拒答时报错。"""
        mock_resp = MagicMock()
        mock_choice = MagicMock(finish_reason="stop")
        mock_choice.message.refusal = "敏感内容触发合规拦截"
        mock_resp.choices = [mock_choice]
        with pytest.raises(ValueError, match="MiMo API 拒绝响应"):
            parse_mimo_response(mock_resp)

    def test_null_or_invalid_type_content(self):
        """测试 content 为 None 或非字符串类型时明确报错，避免静默变空白字幕。"""
        mock_resp = MagicMock()
        mock_choice = MagicMock(finish_reason="stop")
        mock_choice.message.refusal = None
        mock_resp.choices = [mock_choice]

        # content 为 None
        mock_choice.message.content = None
        with pytest.raises(ValueError, match="content 为 null"):
            parse_mimo_response(mock_resp)

        # content 为字典类型
        mock_choice.message.content = {"invalid": "dict"}
        with pytest.raises(ValueError, match="content 类型无效"):
            parse_mimo_response(mock_resp)


# =====================================================================
# 3. 真实 OpenAI SDK 序列化与 httpx.MockTransport 请求拦截
# =====================================================================
class TestOpenAISDKSerializationWithMockTransport:
    """使用 httpx.MockTransport 拦截实际 HTTP 请求，验证 SDK 序列化与自定义路由。"""

    def test_openai_sdk_serialization_to_custom_path(self, tmp_path):
        """验证发送到自定义 Base URL 路径的 HTTP POST 请求体及协议参数。"""
        test_wav = tmp_path / "sdk_test.wav"
        create_test_wav(test_wav, duration_sec=1.0)

        captured_requests = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            resp_body = {
                "id": "chatcmpl-test-123",
                "object": "chat.completion",
                "created": 1720000000,
                "model": "mimo-v2.5-asr",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "通过 MockTransport 识别成功",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx.Response(200, json=resp_body)

        transport = httpx.MockTransport(mock_handler)
        custom_base = "https://custom.gateway.internal/channel-b/v1"
        mock_client = OpenAI(
            base_url=custom_base,
            api_key="sk-test-secret-key-mock",
            http_client=httpx.Client(transport=transport),
        )

        with patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions") as mock_vad:
            mock_vad.return_value = [(0.0, 1.0)]

            asr = MiMoASR(
                audio_input=str(test_wav),
                base_url=custom_base,
                api_key="sk-test-secret-key-mock",
                model="mimo-v2.5-asr",
                language="zh",
                vad_filter=True,
                client=mock_client,
            )
            result = asr.run()

        assert len(result.segments) == 1
        assert result.segments[0].text == "通过 MockTransport 识别成功"

        # 检查捕获到的真实 HTTP 请求
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.url.path == "/channel-b/v1/chat/completions"
        assert req.method == "POST"

        # 检查反序列化请求体
        req_json = json.loads(req.content.decode("utf-8"))
        assert req_json["model"] == "mimo-v2.5-asr"
        assert req_json["asr_options"]["language"] == "zh"
        msg_content = req_json["messages"][0]["content"][0]
        assert msg_content["type"] == "input_audio"
        assert msg_content["input_audio"]["data"].startswith("data:audio/wav;base64,")

    def test_mock_transport_handles_malformed_response_cleanly(self, tmp_path):
        """当服务端返回 null content 畸形数据时，验证 MiMoASR 清洗并报错，不静默成功。"""
        test_wav = tmp_path / "sdk_bad.wav"
        create_test_wav(test_wav, duration_sec=1.0)

        def mock_bad_handler(request: httpx.Request) -> httpx.Response:
            resp_body = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "stop",
                    }
                ]
            }
            return httpx.Response(200, json=resp_body)

        transport = httpx.MockTransport(mock_bad_handler)
        mock_client = OpenAI(
            base_url="https://api.xiaomimimo.com/v1",
            api_key="sk-test-secret",
            http_client=httpx.Client(transport=transport),
        )

        with patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions") as mock_vad:
            mock_vad.return_value = [(0.0, 1.0)]

            asr = MiMoASR(
                audio_input=str(test_wav),
                base_url="https://api.xiaomimimo.com/v1",
                api_key="sk-test-secret",
                client=mock_client,
            )
            with pytest.raises(RuntimeError, match="content 为 null"):
                asr.run()


# =====================================================================
# 4. 探针功能与异常脱敏测试
# =====================================================================
class TestProbeAndErrorSanitization:
    """测试探针测试与敏感信息清洗。"""

    def test_sanitize_error_masks_api_key(self):
        secret_key = "sk-sensitive-mimo-key-99999"
        raw_error = f"Failed to authenticate with bearer {secret_key}"
        cleaned = _sanitize_error(raw_error, secret_key)
        assert secret_key not in cleaned
        assert "***" in cleaned

    def test_probe_empty_response_is_valid_success(self):
        """测试探针对静音返回空文本时判定为连接成功。"""
        with patch("videocaptioner.core.llm.check_mimo.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_resp = MagicMock()
            mock_choice = MagicMock(finish_reason="stop")
            mock_choice.message.refusal = None
            mock_choice.message.content = ""  # 静音返回空
            mock_resp.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_resp

            ok, msg = check_mimo_connection(
                base_url="https://api.xiaomimimo.com/v1",
                api_key="sk-valid-key",
            )
            assert ok is True
            assert "接口连接成功" in msg

    def test_probe_invalid_schema_fails(self):
        """测试探针收到异常 schema 时判定为失败。"""
        with patch("videocaptioner.core.llm.check_mimo.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_resp = MagicMock(choices=[])
            mock_client.chat.completions.create.return_value = mock_resp

            ok, msg = check_mimo_connection(
                base_url="https://api.xiaomimimo.com/v1",
                api_key="sk-valid-key",
            )
            assert ok is False
            assert "数据解析错误" in msg

    def test_probe_success_does_not_echo_server_untrusted_text(self):
        """测试探针成功时返回中立成功信息，不回显不可信的服务端返回文本。"""
        with patch("videocaptioner.core.llm.check_mimo.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_resp = MagicMock()
            mock_choice = MagicMock(finish_reason="stop")
            mock_choice.message.refusal = None
            mock_choice.message.content = "恶意注入或不安全内容"
            mock_resp.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_resp

            ok, msg = check_mimo_connection(
                base_url="https://api.xiaomimimo.com/v1",
                api_key="sk-test",
            )
            assert ok is True
            assert "接口连接成功" in msg
            assert "恶意注入" not in msg

    def test_probe_constructor_error_sanitized(self):
        """测试在 OpenAI 客户端构造器抛出异常时依然保持 tuple 契约并脱敏。"""
        secret = "sk-leaky-constructor-token-777"
        with patch("videocaptioner.core.llm.check_mimo.OpenAI") as mock_openai:
            mock_openai.side_effect = RuntimeError(f"Constructor failed with token {secret}")
            ok, msg = check_mimo_connection(
                base_url="https://api.xiaomimimo.com/v1",
                api_key=secret,
            )
            assert ok is False
            assert secret not in msg
            assert "***" in msg

    def test_tiny_probe_wav_format(self):
        """测试探针 WAV 文件基本结构。"""
        wav_bytes = generate_tiny_probe_wav()
        assert len(wav_bytes) > 44
        assert wav_bytes.startswith(b"RIFF")


# =====================================================================
# 5. 缓存键验证与前置凭据校验
# =====================================================================
class TestCacheKeyAndCredentialPrecheck:
    """测试缓存键规范与前置凭据校验。"""

    def test_cache_key_includes_endpoint_hash_and_duration_without_key(self, tmp_path):
        wav_path = tmp_path / "cache_test.wav"
        create_test_wav(wav_path, duration_sec=1.0)

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="sk-secret-do-not-leak",
            model="mimo-v2.5-asr",
            language="zh",
            vad_filter=True,
            max_chunk_duration=25.0,
        )

        cache_key = asr._get_key()
        assert "sk-secret-do-not-leak" not in cache_key
        expected_hash = hashlib.sha256("https://api.xiaomimimo.com/v1".encode()).hexdigest()[:12]
        assert f"ep_{expected_hash}" in cache_key
        assert "chunk_25.0" in cache_key

    def test_run_validates_credentials_before_cache(self, tmp_path):
        """当 Base URL 或 API Key 为空时，run() 必须在执行或读取缓存前抛出 ValueError。"""
        wav_path = tmp_path / "cred_test.wav"
        create_test_wav(wav_path, duration_sec=1.0)

        asr_no_key = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="",
        )
        with pytest.raises(ValueError, match="请先在设置中配置 MiMo API Key"):
            asr_no_key.run()

        asr_no_url = MiMoASR(
            audio_input=str(wav_path),
            base_url="",
            api_key="sk-test",
        )
        with pytest.raises(ValueError, match="请先在设置中配置有效的 MiMo API Base URL"):
            asr_no_url.run()


# =====================================================================
# 6. 多点取消机制严谨性测试
# =====================================================================
class TestCancellationHandling:
    """测试任务取消机制，严禁部分成功写入缓存。"""

    @patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions")
    def test_cancel_before_run(self, mock_vad, tmp_path):
        """测试在 run() 执行前取消直接抛出异常。"""
        wav_path = tmp_path / "cancel_pre.wav"
        create_test_wav(wav_path, duration_sec=5.0)

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="sk-test",
        )
        asr.cancel()
        with pytest.raises(RuntimeError, match="任务已被用户取消"):
            asr.run()

    @patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions")
    def test_cancel_after_vad_silence(self, mock_vad, tmp_path):
        """测试即使 VAD 返回全静音，若已取消也必须抛出异常，不能返回成功。"""
        wav_path = tmp_path / "cancel_vad_silence.wav"
        create_test_wav(wav_path, duration_sec=5.0)

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="sk-test",
            vad_filter=True,
        )

        def vad_side_effect(*args, **kwargs):
            asr.cancel()
            return []

        mock_vad.side_effect = vad_side_effect
        with pytest.raises(RuntimeError, match="任务已被用户取消"):
            asr.run()

    @patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions")
    def test_cancel_between_chunks_no_partial_result(self, mock_vad, tmp_path):
        """测试在多分块中间取消时抛出异常，不返回部分结果。"""
        wav_path = tmp_path / "cancel_mid.wav"
        create_test_wav(wav_path, duration_sec=20.0)
        mock_vad.return_value = [(0.0, 5.0), (10.0, 15.0)]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="第一块文本", refusal=None))]
        mock_client.chat.completions.create.return_value = mock_resp

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="sk-test",
            client=mock_client,
        )

        # 在完成第一块后触发取消
        orig_encode = asr.encode_audio_to_base64
        call_count = 0

        def encode_hook(wav_bytes):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                asr.cancel()
            return orig_encode(wav_bytes)

        asr.encode_audio_to_base64 = encode_hook

        with pytest.raises(RuntimeError, match="任务已被用户取消"):
            asr.run()


# =====================================================================
# 7. VAD 全静音 0 API 调用与原始时间轴保持 (1-3s, 10-12s)
# =====================================================================
class TestVadZeroCallAndTimelineOffsets:
    """测试全静音 0 API 请求与原始时间戳偏移保持。"""

    @patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions")
    def test_vad_all_silence_makes_zero_api_calls(self, mock_vad, tmp_path):
        """测试全静音检测结果产生 0 次 API 调用，返回空 ASRData。"""
        wav_path = tmp_path / "silence.wav"
        create_test_wav(wav_path, duration_sec=5.0)
        mock_vad.return_value = []

        mock_client = MagicMock()

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="test-key",
            vad_filter=True,
            client=mock_client,
        )
        result = asr.run()

        assert mock_client.chat.completions.create.call_count == 0
        assert len(result.segments) == 0

    @patch("videocaptioner.core.asr.mimo_asr.detect_speech_regions")
    def test_timeline_preservation_with_silence_removal(self, mock_vad, tmp_path):
        """测试在静音切除后，[1.0s-3.0s] 与 [10.0s-12.0s] 严格映射回原始时间轴，无平移。"""
        wav_path = tmp_path / "timeline.wav"
        create_test_wav(wav_path, duration_sec=15.0)

        mock_vad.return_value = [(1.0, 3.0), (10.0, 12.0)]
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            mock_res = MagicMock()
            if mock_client.chat.completions.create.call_count == 1:
                mock_res.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="第一句话", refusal=None))]
            else:
                mock_res.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="第二句话", refusal=None))]
            return mock_res

        mock_client.chat.completions.create.side_effect = side_effect

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="test-key",
            vad_filter=True,
            client=mock_client,
        )
        result = asr.run()

        assert len(result.segments) == 2
        seg1, seg2 = result.segments[0], result.segments[1]

        assert seg1.text == "第一句话"
        assert seg1.start_time == 1000
        assert seg1.end_time == 3000

        # 第二句未发生任何平移
        assert seg2.text == "第二句话"
        assert seg2.start_time == 10000
        assert seg2.end_time == 12000


# =====================================================================
# 8. VAD 关闭时 61s 音频边界切块与长语音拆分
# =====================================================================
class TestVadDisabledChunkingAndLongSpeechSplit:
    """测试 VAD 关闭时的 25s 边界切片与长语音拆分。"""

    def test_vad_disabled_61s_sequential_chunks(self, tmp_path):
        """测试 61 秒音频在 VAD 关闭时被精准拆分为 [0-25s], [25-50s], [50-61s] 3 个请求。"""
        wav_path = tmp_path / "stream_61s.wav"
        create_test_wav(wav_path, duration_sec=61.0)

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="分块内容", refusal=None))]
        mock_client.chat.completions.create.return_value = mock_resp

        asr = MiMoASR(
            audio_input=str(wav_path),
            base_url="https://api.xiaomimimo.com/v1",
            api_key="test-key",
            vad_filter=False,
            max_chunk_duration=25.0,
            client=mock_client,
        )
        result = asr.run()

        assert mock_client.chat.completions.create.call_count == 3
        assert len(result.segments) == 3
        assert (result.segments[0].start_time, result.segments[0].end_time) == (0, 25000)
        assert (result.segments[1].start_time, result.segments[1].end_time) == (25000, 50000)
        assert (result.segments[2].start_time, result.segments[2].end_time) == (50000, 61000)

    def test_merge_speech_regions_splits_long_continuous_speech(self):
        """测试单个长区间 (>25s) 被拆分成不超过 max_chunk_duration 的子区间。"""
        regions = [(0.0, 55.0)]
        merged = merge_speech_regions_into_chunks(regions, max_chunk_duration=25.0)
        assert len(merged) == 3
        assert merged[0] == (0.0, 25.0)
        assert merged[1] == (25.0, 50.0)
        assert merged[2] == (50.0, 55.0)


# =====================================================================
# 9. 单块 Base64 >10MB 大小限制检测
# =====================================================================
class TestBase64SizeLimit:
    """测试 Base64 大小限制检测。"""

    def test_base64_size_limit_validation(self):
        small_audio = b"RIFF" + b"\x00" * 1000
        b64 = MiMoASR.encode_audio_to_base64(small_audio)
        assert isinstance(b64, str)

        with patch("base64.b64encode") as mock_b64:
            # 模拟生成 11 MB 的 base64 字符串
            mock_b64.return_value = b"x" * (11 * 1024 * 1024)
            with pytest.raises(ValueError, match="超过 MiMo API 10 MB 限制"):
                MiMoASR.encode_audio_to_base64(small_audio)


# =====================================================================
# 10. transcribe() 绕过 optimize_timing() 边界保持测试
# =====================================================================
class TestTranscribeTimingBypass:
    """测试 transcribe() 针对 MiMo 绕过 optimize_timing() 以保持精确分块边界。"""

    @patch("videocaptioner.core.asr.transcribe._create_asr_instance")
    def test_mimo_timing_not_modified_by_optimize_timing(self, mock_create, tmp_path):
        wav_path = tmp_path / "gap_test.wav"
        create_test_wav(wav_path, duration_sec=4.0)

        mock_asr = MagicMock()
        initial_data = ASRData(
            [
                ASRDataSeg("第一句", 0, 1000),
                ASRDataSeg("第二句", 1050, 2000),  # gap = 50ms
            ]
        )
        mock_asr.run.return_value = initial_data
        mock_create.return_value = mock_asr

        config = TranscribeConfig(
            transcribe_model=TranscribeModelEnum.MIMO_ASR,
            need_word_time_stamp=False,
            mimo_api_base="https://api.xiaomimimo.com/v1",
            mimo_api_key="test-key",
        )

        result = transcribe(str(wav_path), config)
        assert result.segments[0].end_time == 1000
        assert result.segments[1].start_time == 1050


# =====================================================================
# 11. 独立语言配置隔离与 TaskFactory 映射测试
# =====================================================================
class TestMiMoLanguageConfigIsolation:
    """测试 MiMo 独立语言枚举与全局语言的完全隔离。"""

    def test_unrelated_global_japanese_does_not_corrupt_mimo(self, tmp_path):
        test_file = tmp_path / "sample.mp4"
        test_file.touch()

        orig_global_lang = cfg.transcribe_language.value
        orig_mimo_lang = cfg.mimo_transcribe_language.value
        orig_model = cfg.transcribe_model.value

        try:
            cfg.transcribe_language.value = TranscribeLanguageEnum.JAPANESE
            cfg.transcribe_model.value = TranscribeModelEnum.MIMO_ASR
            cfg.mimo_transcribe_language.value = MiMoLanguageEnum.CHINESE

            task = TaskFactory.create_transcribe_task(str(test_file))
            assert task.transcribe_config.transcribe_language == "zh"
            assert task.transcribe_config.need_word_time_stamp is False

            cfg.mimo_transcribe_language.value = MiMoLanguageEnum.AUTO
            task_auto = TaskFactory.create_transcribe_task(str(test_file))
            assert task_auto.transcribe_config.transcribe_language == "auto"

            cfg.mimo_transcribe_language.value = MiMoLanguageEnum.ENGLISH
            task_en = TaskFactory.create_transcribe_task(str(test_file))
            assert task_en.transcribe_config.transcribe_language == "en"
        finally:
            cfg.transcribe_language.value = orig_global_lang
            cfg.mimo_transcribe_language.value = orig_mimo_lang
            cfg.transcribe_model.value = orig_model


# =====================================================================
# 12. 配置序列化 Roundtrip 测试 (隔离验证)
# =====================================================================
class TestConfigSerializationRoundtrip:
    """测试 MiMo 语言配置项的序列化与反序列化，不写入用户真实文件。"""

    def test_mimo_language_serializer_roundtrip(self):
        serializer = cfg.mimo_transcribe_language.serializer
        for enum_val in MiMoLanguageEnum:
            serialized = serializer.serialize(enum_val)
            deserialized = serializer.deserialize(serialized)
            assert deserialized == enum_val


# =====================================================================
# 13. 本地 Silero VAD 环境检测与真实 Fixture 测试
# =====================================================================
class TestSileroVadDetection:
    """测试本地 Silero VAD 检测。"""

    def test_check_vad_environment_fast_check(self):
        """测试 VAD 环境检测能够快速返回结果。"""
        is_ready, msg = check_vad_environment()
        assert isinstance(is_ready, bool)
        assert isinstance(msg, str)

    def test_check_vad_missing_packages_returns_false(self, tmp_path):
        """当伪造一个缺少包的 Python 路径时，验证不会随意 fallback 返回 True。"""
        fake_py = tmp_path / "Scripts" / "python.exe"
        fake_py.parent.mkdir(parents=True)
        fake_py.touch()

        # 没有 site-packages
        ready, msg = check_vad_environment(fake_py)
        assert ready is False
        assert "site-packages" in msg

    def test_real_silero_vad_execution_if_available(self):
        """如果本地存在独立运行时，基于 zh.mp3 fixture 验证实际检测。"""
        is_ready, msg = check_vad_environment()
        if not is_ready:
            pytest.skip(f"VAD 环境不可用，跳过真实执行: {msg}")

        fixture_audio = Path("tests/fixtures/audio/zh.mp3")
        assert fixture_audio.exists()
        regions = detect_speech_regions(str(fixture_audio))
        assert len(regions) >= 1
        assert regions[0][0] >= 0.0
        assert regions[0][1] > regions[0][0]


# =====================================================================
# 14. Qt 线程生命周期与 Worker 验证 (QCoreApplication ONLY)
# =====================================================================
class TestMiMoConnectionThreadLifecycle:
    """使用 QCoreApplication 测试 MiMoConnectionThread 的信号与退出清理。"""

    @pytest.fixture(scope="class")
    def qcore_app(self):
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        return app

    def test_worker_auto_registration_and_automatic_cleanup(self, qcore_app):
        """测试在 QCoreApplication 下，Worker 自动加入全局注册表，完成时自动清理移除。"""
        worker = MiMoConnectionThread(
            base_url="https://api.xiaomimimo.com/v1",
            api_key="sk-test",
            model="mimo-v2.5-asr",
        )
        # 验证实例化即自动注册到全局活跃集合，无需 Caller 手动管理
        assert worker in _ACTIVE_MIMO_WORKERS

        received_results = []
        finished_called = []

        worker.result_ready.connect(lambda s, m: received_results.append((s, m)))
        worker.finished.connect(lambda: finished_called.append(True))

        with patch("videocaptioner.ui.components.MiMoASRSettingWidget.check_mimo_connection") as mock_chk:
            mock_chk.return_value = (True, "连接探针成功")
            worker.start()
            worker.wait(5000)
            qcore_app.processEvents()

        assert len(received_results) == 1
        assert received_results[0][0] is True
        assert len(finished_called) == 1

        # 验证模块级自动完成生命周期清理，worker 已经安全从全局活跃集合中移除
        assert worker not in _ACTIVE_MIMO_WORKERS

    def test_import_before_app_creation_connects_about_to_quit_and_cleans_up(self):
        """严格验证：在创建 QCoreApplication 之前 import 模块，后续创建 App 并运行 worker 能正常绑定 aboutToQuit 与清理。"""
        import subprocess
        import sys

        code = """
import sys
# 1. 在创建 QCoreApplication 之前导入模块
from videocaptioner.ui.components.MiMoASRSettingWidget import (
    MiMoConnectionThread,
    _ACTIVE_MIMO_WORKERS,
    _cleanup_all_mimo_workers,
)
from PyQt5.QtCore import QCoreApplication
from unittest.mock import patch

# 验证此时应用实例尚未创建
assert QCoreApplication.instance() is None

# 2. 模拟 videocaptioner/ui/main.py 之后才创建 Application
app = QCoreApplication([])
assert QCoreApplication.instance() is not None

# 3. 实例化 worker，验证自动连接 aboutToQuit 与自动加入活跃注册表
worker = MiMoConnectionThread("https://api.xiaomimimo.com/v1", "sk-test", "mimo-v2.5-asr")
assert worker in _ACTIVE_MIMO_WORKERS

# 4. 模拟 probe 运行与自动回收
results = []
worker.result_ready.connect(lambda s, m: results.append((s, m)))
with patch("videocaptioner.ui.components.MiMoASRSettingWidget.check_mimo_connection", return_value=(True, "OK")):
    worker.start()
    worker.wait(3000)
    app.processEvents()

assert len(results) == 1 and results[0][0] is True
assert worker not in _ACTIVE_MIMO_WORKERS

# 5. 验证创建新 worker 后触发 aboutToQuit 会安全调用 worker.stop()
worker2 = MiMoConnectionThread("https://api.xiaomimimo.com/v1", "sk-test", "mimo-v2.5-asr")
assert worker2 in _ACTIVE_MIMO_WORKERS
app.aboutToQuit.emit()
assert worker2._is_stopped is True
worker2.stop()
_ACTIVE_MIMO_WORKERS.discard(worker2)
print("ISOLATED_LIFECYCLE_SUCCESS")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert "ISOLATED_LIFECYCLE_SUCCESS" in proc.stdout

    def test_vad_subprocess_timeout_is_600_seconds(self):
        """验证 VAD 子进程超时设定为 >=600 秒且错误提示清晰。"""
        from videocaptioner.core.asr import mimo_vad

        src = inspect.getsource(mimo_vad.detect_speech_regions)
        assert "timeout=600" in src
        assert "600 秒" in src
