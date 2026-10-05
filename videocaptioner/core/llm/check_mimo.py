"""Xiaomi MiMo ASR API connection check and response parsing utilities."""

import base64
import io
import wave
from typing import Any, Optional
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from videocaptioner.core.llm.mimo_rate_limit import (
    DEFAULT_MIMO_RPM,
    DEFAULT_MIMO_TPM,
    get_mimo_limiter,
    send_with_mimo_limits,
)


def validate_and_normalize_mimo_base_url(base_url: str) -> str:
    """Validate and normalize user-supplied MiMo API Base URL.

    Rules:
    - Must be empty or start with http:// or https://
    - Must contain a valid hostname
    - Must NOT contain credentials (username/password), query parameters, or fragments
    - Strips whitespace and trailing slashes
    - Preserves custom versioned base paths (e.g. token plan or standard channel)

    Returns:
        Normalized URL string, or "" if input is empty/whitespace.

    Raises:
        ValueError: If URL format violates security or structural constraints.
    """
    raw = (base_url or "").strip()
    if not raw:
        return ""

    if any(c.isspace() or ord(c) < 32 for c in raw):
        raise ValueError("Base URL 包含非法空白或控制字符")

    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("Base URL 协议无效，必须以 http:// 或 https:// 开头")

    if not parsed.netloc:
        raise ValueError("Base URL 必须包含有效的主机名 (Host)")

    try:
        if parsed.port is not None and not (1 <= parsed.port <= 65535):
            raise ValueError("Base URL 端口号超出合法范围 (1-65535)")
    except ValueError as e:
        raise ValueError(f"Base URL 端口号无效: {e}")

    if parsed.username or parsed.password:
        raise ValueError("Base URL 不能包含用户名或密码等凭据信息")

    if parsed.query:
        raise ValueError("Base URL 不能包含查询参数 (?...)")

    if parsed.fragment:
        raise ValueError("Base URL 不能包含片段标识符 (#...)")

    # 验证通过后直接返回原始 URL 去除末尾斜杠，完整保留 IPv6 方括号与自定义路径
    return raw.rstrip("/")


def normalize_mimo_base_url(base_url: str) -> str:
    """Backward-compatible alias for validate_and_normalize_mimo_base_url."""
    return validate_and_normalize_mimo_base_url(base_url)


def generate_tiny_probe_wav() -> bytes:
    """Generate an in-memory 0.1s 16kHz 16-bit mono PCM silent WAV probe."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        # 1600 frames = 0.1s at 16000 Hz, 2 bytes per sample
        wav_file.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


def _sanitize_error(error_msg: str, api_key: Optional[str] = None) -> str:
    """Sanitize error messages to avoid revealing API keys or sensitive tokens."""
    sanitized = str(error_msg or "")
    if api_key and api_key.strip():
        sanitized = sanitized.replace(api_key.strip(), "***")
    return sanitized


def parse_mimo_response(response: Any) -> str:
    """Parse and strictly validate a MiMo-ASR chat.completions response.

    Rules:
    - choices must be a non-empty list.
    - choice.finish_reason == 'length' indicates truncation and must fail.
    - choice.message must exist and not be None.
    - choice.message.refusal must be empty/None.
    - choice.message.content must be a valid string (empty string "" is valid silence).
      Missing, None, or wrong type must raise ValueError.

    Returns:
        Stripped recognition content string.

    Raises:
        ValueError: If response schema is malformed or invalid.
    """
    if not response or not hasattr(response, "choices") or not response.choices:
        raise ValueError("MiMo API 返回的数据结构无效: choices 为空")

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise ValueError("MiMo API 输出由于长度超限被截断 (finish_reason=length)")

    message = getattr(choice, "message", None)
    if message is None:
        raise ValueError("MiMo API 返回的数据结构无效: 缺少 message 字段")

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"MiMo API 拒绝响应: {refusal}")

    content = getattr(message, "content", None)
    if content is None:
        raise ValueError("MiMo API 返回的消息 content 为 null，响应结构不符合预期")

    if not isinstance(content, str):
        raise ValueError(f"MiMo API 返回的消息 content 类型无效: {type(content).__name__}")

    return content.strip()


def check_mimo_connection(
    base_url: str,
    api_key: str,
    model: str = "mimo-v2.5-asr",
    client: Optional[OpenAI] = None,
    rpm: int = DEFAULT_MIMO_RPM,
    tpm: int = DEFAULT_MIMO_TPM,
    cancelled=lambda: False,
) -> tuple[bool, str]:
    """Test MiMo ASR API connectivity using the real speech recognition endpoint.

    Uses a tiny built-in silent WAV probe to test the recognition endpoint.
    Handles empty recognition text for silence as a successful response,
    while treating authentication/network/schema issues as errors.
    """
    clean_key = (api_key or "").strip()
    target_model = (model or "").strip() or "mimo-v2.5-asr"

    try:
        norm_url = validate_and_normalize_mimo_base_url(base_url)
    except ValueError as e:
        return False, _sanitize_error(str(e), clean_key)

    if not norm_url:
        return False, "请输入 MiMo API Base URL"
    if not clean_key:
        return False, "请输入 MiMo API Key"

    local_client = client
    should_close_client = False

    try:
        if local_client is None:
            local_client = OpenAI(
                base_url=norm_url,
                api_key=clean_key,
                timeout=10,
                max_retries=0,
            )
            should_close_client = True

        probe_wav = generate_tiny_probe_wav()
        b64_audio = base64.b64encode(probe_wav).decode("ascii")

        def send_request():
            return local_client.chat.completions.create(
                model=target_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": f"data:audio/wav;base64,{b64_audio}"},
                            }
                        ],
                    }
                ],
                extra_body={"asr_options": {"language": "auto"}},
            )

        response = send_with_mimo_limits(
            send_request,
            get_mimo_limiter(norm_url, target_model),
            0.1,
            rpm=rpm,
            tpm=tpm,
            cancelled=cancelled,
            max_retries=0,
        )
        parse_mimo_response(response)
        # Do not echo untrusted server output or credentials; return clean success
        return True, "MiMo-ASR 接口连接成功！（识别端点响应正常）"

    except APIConnectionError:
        return False, "API 连接失败，请检查网络连接或 Base URL 是否可访问"
    except RateLimitError:
        return (
            False,
            "MiMo 服务可达，但当前请求被限流 (429)。请稍后测试；同账号的其他程序也会占用额度。",
        )
    except AuthenticationError:
        return False, "身份验证失败 (401)，请检查 API Key 是否有效"
    except NotFoundError:
        return False, "请求地址未找到 (404)，请检查 Base URL 是否正确"
    except BadRequestError as e:
        return False, f"请求参数错误 (Bad Request): {_sanitize_error(str(e), clean_key)}"
    except OpenAIError as e:
        return False, f"API 调用错误: {_sanitize_error(str(e), clean_key)}"
    except ValueError as e:
        return False, f"数据解析错误: {_sanitize_error(str(e), clean_key)}"
    except Exception as e:
        return False, f"测试异常: {_sanitize_error(str(e), clean_key)}"
    finally:
        if should_close_client and local_client is not None:
            try:
                local_client.close()
            except Exception:
                pass
