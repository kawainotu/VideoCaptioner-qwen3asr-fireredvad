# coding:utf-8
from enum import Enum

from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QColor
from qfluentwidgets import (
    BoolValidator,
    ConfigItem,
    ConfigSerializer,
    EnumSerializer,
    FolderValidator,
    OptionsConfigItem,
    OptionsValidator,
    QConfig,
    RangeConfigItem,
    RangeValidator,
    Theme,
    qconfig,
)

from videocaptioner.config import SETTINGS_PATH, WORK_PATH
from videocaptioner.core.asr.qwen3_models import (
    DEFAULT_QWEN3_ASR_MODEL_KEY,
    QWEN3_ASR_MODELS,
)
from videocaptioner.core.asr.qwen3_vad_models import (
    DEFAULT_QWEN3_VAD_MODEL_KEY,
    QWEN3_VAD_MODELS,
)
from videocaptioner.core.entities import (
    FasterWhisperModelEnum,
    LLMServiceEnum,
    SubtitleLayoutEnum,
    SubtitleRenderModeEnum,
    TranscribeLanguageEnum,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
    TranslatorServiceEnum,
    VadMethodEnum,
    VideoQualityEnum,
    WhisperModelEnum,
)
from videocaptioner.core.qwen3_vad_defaults import (
    FIRERED_VAD_DEFAULTS,
    LEGACY_FIRERED_VAD_DEFAULTS,
)
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.core.utils.platform_utils import get_available_transcribe_models


class Language(Enum):
    """软件语言"""

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
    """Language serializer"""

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


class PlatformAwareTranscribeModelValidator(OptionsValidator):
    """平台相关的转录模型验证器，在 macOS 上自动过滤掉 FasterWhisper"""

    def __init__(self):
        # 不调用父类的 __init__，因为我们要自定义 options
        self._options = get_available_transcribe_models()

    @property
    def options(self):
        return self._options

    def validate(self, value):
        return value in self._options

    def correct(self, value):
        return value if self.validate(value) else self._options[0]


class Config(QConfig):
    """应用配置"""

    # LLM配置
    llm_service = OptionsConfigItem(
        "LLM",
        "LLMService",
        LLMServiceEnum.OPENAI,
        OptionsValidator(LLMServiceEnum),
        EnumSerializer(LLMServiceEnum),
    )

    openai_model = ConfigItem("LLM", "OpenAI_Model", "gpt-4o-mini")
    openai_api_key = ConfigItem("LLM", "OpenAI_API_Key", "")
    openai_api_base = ConfigItem("LLM", "OpenAI_API_Base", "https://api.openai.com/v1")

    silicon_cloud_model = ConfigItem("LLM", "SiliconCloud_Model", "gpt-4o-mini")
    silicon_cloud_api_key = ConfigItem("LLM", "SiliconCloud_API_Key", "")
    silicon_cloud_api_base = ConfigItem(
        "LLM", "SiliconCloud_API_Base", "https://api.siliconflow.cn/v1"
    )

    deepseek_model = ConfigItem("LLM", "DeepSeek_Model", "deepseek-chat")
    deepseek_api_key = ConfigItem("LLM", "DeepSeek_API_Key", "")
    deepseek_api_base = ConfigItem(
        "LLM", "DeepSeek_API_Base", "https://api.deepseek.com/v1"
    )

    ollama_model = ConfigItem("LLM", "Ollama_Model", "llama2")
    ollama_api_key = ConfigItem("LLM", "Ollama_API_Key", "ollama")
    ollama_api_base = ConfigItem("LLM", "Ollama_API_Base", "http://localhost:11434/v1")

    lm_studio_model = ConfigItem("LLM", "LmStudio_Model", "qwen2.5:7b")
    lm_studio_api_key = ConfigItem("LLM", "LmStudio_API_Key", "lmstudio")
    lm_studio_api_base = ConfigItem(
        "LLM", "LmStudio_API_Base", "http://localhost:1234/v1"
    )

    gemini_model = ConfigItem("LLM", "Gemini_Model", "gemini-pro")
    gemini_api_key = ConfigItem("LLM", "Gemini_API_Key", "")
    gemini_api_base = ConfigItem(
        "LLM",
        "Gemini_API_Base",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    chatglm_model = ConfigItem("LLM", "ChatGLM_Model", "glm-4")
    chatglm_api_key = ConfigItem("LLM", "ChatGLM_API_Key", "")
    chatglm_api_base = ConfigItem(
        "LLM", "ChatGLM_API_Base", "https://open.bigmodel.cn/api/paas/v4"
    )

    # ------------------- 翻译配置 -------------------
    translator_service = OptionsConfigItem(
        "Translate",
        "TranslatorServiceEnum",
        TranslatorServiceEnum.BING,
        OptionsValidator(TranslatorServiceEnum),
        EnumSerializer(TranslatorServiceEnum),
    )
    need_reflect_translate = ConfigItem(
        "Translate", "NeedReflectTranslate", False, BoolValidator()
    )
    deeplx_endpoint = ConfigItem("Translate", "DeeplxEndpoint", "")
    batch_size = RangeConfigItem("Translate", "BatchSize", 10, RangeValidator(5, 50))
    thread_num = RangeConfigItem("Translate", "ThreadNum", 10, RangeValidator(1, 50))

    # ------------------- 转录配置 -------------------
    transcribe_model = OptionsConfigItem(
        "Transcribe",
        "TranscribeModel",
        TranscribeModelEnum.BIJIAN,
        PlatformAwareTranscribeModelValidator(),
        EnumSerializer(TranscribeModelEnum),
    )
    transcribe_output_format = OptionsConfigItem(
        "Transcribe",
        "OutputFormat",
        TranscribeOutputFormatEnum.SRT,
        OptionsValidator(TranscribeOutputFormatEnum),
        EnumSerializer(TranscribeOutputFormatEnum),
    )
    transcribe_language = OptionsConfigItem(
        "Transcribe",
        "TranscribeLanguage",
        TranscribeLanguageEnum.AUTO,
        OptionsValidator(TranscribeLanguageEnum),
        EnumSerializer(TranscribeLanguageEnum),
    )

    # ------------------- Whisper Cpp 配置 -------------------
    whisper_model = OptionsConfigItem(
        "Whisper",
        "WhisperModel",
        WhisperModelEnum.TINY,
        OptionsValidator(WhisperModelEnum),
        EnumSerializer(WhisperModelEnum),
    )

    # ------------------- Faster Whisper 配置 -------------------
    faster_whisper_program = ConfigItem(
        "FasterWhisper",
        "Program",
        "faster-whisper-xxl.exe",
    )
    faster_whisper_model = OptionsConfigItem(
        "FasterWhisper",
        "Model",
        FasterWhisperModelEnum.TINY,
        OptionsValidator(FasterWhisperModelEnum),
        EnumSerializer(FasterWhisperModelEnum),
    )
    faster_whisper_model_dir = ConfigItem("FasterWhisper", "ModelDir", "")
    faster_whisper_device = OptionsConfigItem(
        "FasterWhisper", "Device", "cuda", OptionsValidator(["cuda", "cpu"])
    )
    # VAD 参数
    faster_whisper_vad_filter = ConfigItem(
        "FasterWhisper", "VadFilter", True, BoolValidator()
    )
    faster_whisper_vad_threshold = RangeConfigItem(
        "FasterWhisper", "VadThreshold", 0.4, RangeValidator(0, 1)
    )
    faster_whisper_vad_method = OptionsConfigItem(
        "FasterWhisper",
        "VadMethod",
        VadMethodEnum.SILERO_V4,
        OptionsValidator(VadMethodEnum),
        EnumSerializer(VadMethodEnum),
    )
    # 人声提取
    faster_whisper_ff_mdx_kim2 = ConfigItem(
        "FasterWhisper", "FfMdxKim2", False, BoolValidator()
    )
    # 文本处理参数
    faster_whisper_one_word = ConfigItem(
        "FasterWhisper", "OneWord", True, BoolValidator()
    )
    # 提示词
    faster_whisper_prompt = ConfigItem("FasterWhisper", "Prompt", "")

    # ------------------- Qwen3 ASR -------------------
    qwen_asr_model = OptionsConfigItem(
        "QwenASR",
        "Model",
        DEFAULT_QWEN3_ASR_MODEL_KEY,
        OptionsValidator([model.key for model in QWEN3_ASR_MODELS]),
    )
    qwen_asr_device = OptionsConfigItem(
        "QwenASR", "Device", "auto", OptionsValidator(["auto", "cuda", "cpu"])
    )
    qwen_asr_low_memory = ConfigItem(
        "QwenASR", "LowMemory", True, BoolValidator()
    )
    qwen_asr_word_timestamps = ConfigItem(
        "QwenASR", "WordTimestamps", False, BoolValidator()
    )
    qwen_asr_vad_filter = ConfigItem(
        "QwenASR", "VadFilter", True, BoolValidator()
    )
    qwen_asr_vad_model = OptionsConfigItem(
        "QwenASR",
        "VadModel",
        DEFAULT_QWEN3_VAD_MODEL_KEY,
        OptionsValidator([model.key for model in QWEN3_VAD_MODELS]),
    )
    qwen_asr_vad_threshold = RangeConfigItem(
        "QwenASR", "VadThreshold", 0.5, RangeValidator(0, 1)
    )
    qwen_asr_vad_min_speech_ms = RangeConfigItem(
        "QwenASR", "VadMinSpeechMs", 250, RangeValidator(50, 5000)
    )
    qwen_asr_vad_min_silence_ms = RangeConfigItem(
        "QwenASR", "VadMinSilenceMs", 500, RangeValidator(50, 5000)
    )
    qwen_asr_vad_speech_pad_ms = RangeConfigItem(
        "QwenASR", "VadSpeechPadMs", 300, RangeValidator(0, 2000)
    )
    qwen_asr_firered_vad_smooth_window_size = RangeConfigItem(
        "QwenASR",
        "FireRedVadSmoothWindowSize",
        FIRERED_VAD_DEFAULTS.smooth_window_size,
        RangeValidator(1, 101),
    )
    qwen_asr_firered_vad_speech_threshold = RangeConfigItem(
        "QwenASR",
        "FireRedVadSpeechThreshold",
        FIRERED_VAD_DEFAULTS.speech_threshold,
        RangeValidator(0, 1),
    )
    qwen_asr_firered_vad_min_speech_frame = RangeConfigItem(
        "QwenASR",
        "FireRedVadMinSpeechFrame",
        FIRERED_VAD_DEFAULTS.min_speech_frame,
        RangeValidator(1, 5000),
    )
    qwen_asr_firered_vad_max_speech_frame = RangeConfigItem(
        "QwenASR",
        "FireRedVadMaxSpeechFrame",
        FIRERED_VAD_DEFAULTS.max_speech_frame,
        RangeValidator(1, 30000),
    )
    qwen_asr_firered_vad_min_silence_frame = RangeConfigItem(
        "QwenASR",
        "FireRedVadMinSilenceFrame",
        FIRERED_VAD_DEFAULTS.min_silence_frame,
        RangeValidator(1, 5000),
    )
    qwen_asr_firered_vad_merge_silence_frame = RangeConfigItem(
        "QwenASR",
        "FireRedVadMergeSilenceFrame",
        FIRERED_VAD_DEFAULTS.merge_silence_frame,
        RangeValidator(0, 5000),
    )
    qwen_asr_firered_vad_extend_speech_frame = RangeConfigItem(
        "QwenASR",
        "FireRedVadExtendSpeechFrame",
        FIRERED_VAD_DEFAULTS.extend_speech_frame,
        RangeValidator(0, 1000),
    )
    qwen_asr_firered_vad_chunk_max_frame = RangeConfigItem(
        "QwenASR",
        "FireRedVadChunkMaxFrame",
        FIRERED_VAD_DEFAULTS.chunk_max_frame,
        RangeValidator(100, 60000),
    )
    qwen_asr_prompt = ConfigItem("QwenASR", "Prompt", "")

    # ------------------- Whisper API 配置 -------------------
    whisper_api_base = ConfigItem("WhisperAPI", "WhisperApiBase", "")
    whisper_api_key = ConfigItem("WhisperAPI", "WhisperApiKey", "")
    whisper_api_model = OptionsConfigItem("WhisperAPI", "WhisperApiModel", "")
    whisper_api_prompt = ConfigItem("WhisperAPI", "WhisperApiPrompt", "")

    # ------------------- 字幕配置 -------------------
    need_optimize = ConfigItem("Subtitle", "NeedOptimize", False, BoolValidator())
    need_translate = ConfigItem("Subtitle", "NeedTranslate", False, BoolValidator())
    need_split = ConfigItem("Subtitle", "NeedSplit", False, BoolValidator())
    target_language = OptionsConfigItem(
        "Subtitle",
        "TargetLanguage",
        TargetLanguage.SIMPLIFIED_CHINESE,
        OptionsValidator(TargetLanguage),
        EnumSerializer(TargetLanguage),
    )
    max_word_count_cjk = ConfigItem(
        "Subtitle", "MaxWordCountCJK", 28, RangeValidator(8, 100)
    )
    max_word_count_english = ConfigItem(
        "Subtitle", "MaxWordCountEnglish", 20, RangeValidator(8, 100)
    )
    custom_prompt_text = ConfigItem("Subtitle", "CustomPromptText", "")

    # ------------------- 字幕合成配置 -------------------
    soft_subtitle = ConfigItem("Video", "SoftSubtitle", False, BoolValidator())
    need_video = ConfigItem("Video", "NeedVideo", True, BoolValidator())
    video_quality = OptionsConfigItem(
        "Video",
        "VideoQuality",
        VideoQualityEnum.MEDIUM,
        OptionsValidator(VideoQualityEnum),
        EnumSerializer(VideoQualityEnum),
    )
    use_subtitle_style = ConfigItem("Video", "UseSubtitleStyle", False, BoolValidator())

    # ------------------- 字幕样式配置 -------------------
    subtitle_style_name = ConfigItem("SubtitleStyle", "StyleName", "default")
    subtitle_layout = OptionsConfigItem(
        "SubtitleStyle",
        "Layout",
        SubtitleLayoutEnum.TRANSLATE_ON_TOP,
        OptionsValidator(SubtitleLayoutEnum),
        EnumSerializer(SubtitleLayoutEnum),
    )
    subtitle_preview_image = ConfigItem("SubtitleStyle", "PreviewImage", "")

    # 字幕渲染模式
    subtitle_render_mode = OptionsConfigItem(
        "SubtitleStyle",
        "RenderMode",
        SubtitleRenderModeEnum.ROUNDED_BG,
        OptionsValidator(SubtitleRenderModeEnum),
        EnumSerializer(SubtitleRenderModeEnum),
    )

    # 圆角背景模式配置
    rounded_bg_font_name = ConfigItem("RoundedBgStyle", "FontName", "Noto Sans SC")
    rounded_bg_font_size = RangeConfigItem(
        "RoundedBgStyle", "FontSize", 52, RangeValidator(16, 120)
    )
    # 背景色：深灰半透明 (R=25, G=25, B=25, A=200)
    rounded_bg_color = ConfigItem("RoundedBgStyle", "BgColor", "#191919C8")
    rounded_bg_text_color = ConfigItem("RoundedBgStyle", "TextColor", "#FFFFFF")
    rounded_bg_corner_radius = RangeConfigItem(
        "RoundedBgStyle", "CornerRadius", 12, RangeValidator(0, 50)
    )
    rounded_bg_padding_h = RangeConfigItem(
        "RoundedBgStyle", "PaddingH", 28, RangeValidator(4, 100)
    )
    rounded_bg_padding_v = RangeConfigItem(
        "RoundedBgStyle", "PaddingV", 14, RangeValidator(4, 50)
    )
    rounded_bg_margin_bottom = RangeConfigItem(
        "RoundedBgStyle", "MarginBottom", 60, RangeValidator(20, 300)
    )
    rounded_bg_line_spacing = RangeConfigItem(
        "RoundedBgStyle", "LineSpacing", 10, RangeValidator(0, 50)
    )
    rounded_bg_letter_spacing = RangeConfigItem(
        "RoundedBgStyle", "LetterSpacing", 0, RangeValidator(0, 20)
    )

    # ------------------- 保存配置 -------------------
    work_dir = ConfigItem("Save", "Work_Dir", WORK_PATH, FolderValidator())

    # ------------------- 软件页面配置 -------------------
    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", False, BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow",
        "DpiScale",
        "Auto",
        OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]),
        restart=True,
    )
    language = OptionsConfigItem(
        "MainWindow",
        "Language",
        Language.AUTO,
        OptionsValidator(Language),
        LanguageSerializer(),
        restart=True,
    )

    # ------------------- 更新配置 -------------------
    checkUpdateAtStartUp = ConfigItem(
        "Update", "CheckUpdateAtStartUp", True, BoolValidator()
    )

    # ------------------- 缓存配置 -------------------
    cache_enabled = ConfigItem("Cache", "CacheEnabled", True, BoolValidator())


cfg = Config()
cfg.themeMode.value = Theme.DARK
cfg.themeColor.value = QColor("#ff28f08b")
qconfig.load(SETTINGS_PATH, cfg)

_firered_vad_migration = (
    (
        cfg.qwen_asr_firered_vad_smooth_window_size,
        LEGACY_FIRERED_VAD_DEFAULTS.smooth_window_size,
        FIRERED_VAD_DEFAULTS.smooth_window_size,
    ),
    (
        cfg.qwen_asr_firered_vad_speech_threshold,
        LEGACY_FIRERED_VAD_DEFAULTS.speech_threshold,
        FIRERED_VAD_DEFAULTS.speech_threshold,
    ),
    (
        cfg.qwen_asr_firered_vad_min_speech_frame,
        LEGACY_FIRERED_VAD_DEFAULTS.min_speech_frame,
        FIRERED_VAD_DEFAULTS.min_speech_frame,
    ),
    (
        cfg.qwen_asr_firered_vad_max_speech_frame,
        LEGACY_FIRERED_VAD_DEFAULTS.max_speech_frame,
        FIRERED_VAD_DEFAULTS.max_speech_frame,
    ),
    (
        cfg.qwen_asr_firered_vad_min_silence_frame,
        LEGACY_FIRERED_VAD_DEFAULTS.min_silence_frame,
        FIRERED_VAD_DEFAULTS.min_silence_frame,
    ),
    (
        cfg.qwen_asr_firered_vad_merge_silence_frame,
        LEGACY_FIRERED_VAD_DEFAULTS.merge_silence_frame,
        FIRERED_VAD_DEFAULTS.merge_silence_frame,
    ),
    (
        cfg.qwen_asr_firered_vad_extend_speech_frame,
        LEGACY_FIRERED_VAD_DEFAULTS.extend_speech_frame,
        FIRERED_VAD_DEFAULTS.extend_speech_frame,
    ),
    (
        cfg.qwen_asr_firered_vad_chunk_max_frame,
        LEGACY_FIRERED_VAD_DEFAULTS.chunk_max_frame,
        FIRERED_VAD_DEFAULTS.chunk_max_frame,
    ),
)
if all(item.value == old_value for item, old_value, _ in _firered_vad_migration):
    for item, _, new_value in _firered_vad_migration:
        cfg.set(item, new_value)
