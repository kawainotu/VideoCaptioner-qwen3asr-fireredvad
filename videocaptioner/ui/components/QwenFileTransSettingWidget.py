"""Cloud file ASR settings shared by the transcription and settings pages."""

import requests
from PyQt5.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    InfoBar,
    PushSettingCard,
    SettingCard,
    SettingCardGroup,
    SwitchSettingCard,
    TextEdit,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.core.asr.qwen_filetrans_asr import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    QwenFileTransASR,
    validate_recognition_options,
)
from videocaptioner.core.entities import (
    TranscribeLanguageEnum,
    TranscribeModelEnum,
    get_asr_language_capability,
)

from ..common.config import cfg
from .LineEditSettingCard import LineEditSettingCard

_ACTIVE_QWEN_WORKERS = set()
_LAST_APP = None


class _RecognitionTextCard(QWidget):
    """A plain-text editor that saves locally and shows validation in place."""

    def __init__(self, config_item, title, helper, placeholder, parent=None):
        super().__init__(parent)
        self.config_item = config_item
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        self.titleLabel = BodyLabel(title, self)
        self.helperLabel = CaptionLabel(helper, self)
        self.helperLabel.setTextFormat(Qt.PlainText)
        self.helperLabel.setWordWrap(True)
        self.editor = TextEdit(self)
        self.editor.setPlaceholderText(placeholder)
        self.editor.setFixedHeight(100)
        self.validationLabel = CaptionLabel(self)
        self.validationLabel.setTextFormat(Qt.PlainText)
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.helperLabel)
        layout.addWidget(self.editor)
        layout.addWidget(self.validationLabel)
        self.setFixedHeight(240)
        self.set_value(config_item.value)
        self.editor.textChanged.connect(self._save)
        config_item.valueChanged.connect(self.set_value)

    def set_value(self, value):
        if self.editor.toPlainText() == value:
            return
        self.editor.blockSignals(True)
        self.editor.setPlainText(value)
        self.editor.blockSignals(False)

    def _save(self):
        cfg.set(self.config_item, self.editor.toPlainText())


def _stop_workers():
    for worker in tuple(_ACTIVE_QWEN_WORKERS):
        worker.stop()


class QwenFileTransConnectionThread(QThread):
    result_ready = pyqtSignal(bool, str)

    def __init__(self, base_url, api_key, model):
        super().__init__()
        self.settings = (base_url, api_key, model)
        self.asr = None
        self._stopped = False
        self.finished.connect(self._cleanup)

    def start(self, *args, **kwargs):
        global _LAST_APP
        app = QCoreApplication.instance()
        if app is not None and app is not _LAST_APP:
            app.aboutToQuit.connect(_stop_workers)
            _LAST_APP = app
        _ACTIVE_QWEN_WORKERS.add(self)
        super().start(*args, **kwargs)

    def _cleanup(self):
        _ACTIVE_QWEN_WORKERS.discard(self)
        self.deleteLater()

    def stop(self):
        self._stopped = True
        if self.asr is not None:
            self.asr.cancel()

    def run(self):
        if self._stopped:
            return
        try:
            base, key, model = self.settings
            self.asr = QwenFileTransASR("unused.wav", key, base, model)
            if self._stopped:
                self.asr.cancel()
            with requests.Session() as session:
                self.asr.get_upload_policy(session)
            if not self._stopped:
                self.result_ready.emit(
                    True, "API Key 与模型上传权限验证通过；开始转录后会上传音频并产生识别费用。"
                )
        except Exception as error:
            if not self._stopped:
                self.result_ready.emit(
                    False, self.asr._safe_error(error) if self.asr else str(error)
                )
        finally:
            self.asr = None


class QwenFileTransSettingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.destroyed.connect(self.stop)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setting_group = SettingCardGroup(self.tr("Qwen-Audio 云端 ASR"), self)
        self.base_url_card = LineEditSettingCard(
            cfg.qwen_filetrans_api_base,
            FIF.LINK,
            self.tr("API Base URL"),
            self.tr("填写文件转录 API 基础地址，以 /api/v1 结尾"),
            DEFAULT_BASE_URL,
            self,
        )
        self.api_key_card = LineEditSettingCard(
            cfg.qwen_filetrans_api_key,
            FIF.FINGERPRINT,
            self.tr("API Key"),
            self.tr("开始转录时将音频上传到千问临时存储（48 小时有效）"),
            "",
            self,
        )
        self.api_key_card.lineEdit.setEchoMode(QLineEdit.Password)
        self.model_card = LineEditSettingCard(
            cfg.qwen_filetrans_model,
            FIF.ROBOT,
            self.tr("模型"),
            self.tr("模型须支持异步文件转录与临时 OSS 上传"),
            DEFAULT_MODEL,
            self,
        )
        self.language_card = SettingCard(
            FIF.LANGUAGE, self.tr("源语言"), self.tr("自动识别或提供语言提示"), self
        )
        self.language_card.comboBox = ComboBox(self.language_card)
        capability = get_asr_language_capability(TranscribeModelEnum.QWEN_FILETRANS)
        for language in [TranscribeLanguageEnum.AUTO] + capability.supported_languages:
            self.language_card.comboBox.addItem(language.value, userData=language)
        self.language_card.hBoxLayout.addWidget(self.language_card.comboBox, 0, Qt.AlignRight)
        self.language_card.hBoxLayout.addSpacing(16)
        self.word_card = SwitchSettingCard(
            FIF.FONT,
            self.tr("词级时间戳"),
            self.tr("使用服务返回的词时间；未提供时保留句子时间"),
            configItem=cfg.qwen_filetrans_word_timestamps,
            parent=self,
        )
        self.hotwords_card = _RecognitionTextCard(
            cfg.qwen_filetrans_hotwords,
            self.tr("自定义热词"),
            self.tr(
                "每行一个词，默认权重 4；可写“词条 | 4”。权重仅限 1–5 或 50，50 为超级热词。\n"
                "最多 2000 条；含中文、重音字母等字符的词条最多 15 字符，纯英文最多 7 个单词；超级热词最多 50 条。"
            ),
            "斯蒂格勒\n第三持存 | 5\nVideoCaptioner | 4",
            self,
        )
        self.vocabulary_id_card = LineEditSettingCard(
            cfg.qwen_filetrans_vocabulary_id,
            FIF.DOCUMENT,
            self.tr("已有云端词表 ID（可选）"),
            self.tr("填入预先创建的词表 ID；与上方热词合并使用"),
            "vocab-...",
            self,
        )
        self.context_card = _RecognitionTextCard(
            cfg.qwen_filetrans_context,
            self.tr("上下文增强"),
            self.tr(
                "最多 400 字符。说明录音主题并列出人名、术语的准确写法；保留音频中的原文词形。\n"
                "例如：录音讨论斯蒂格勒的第三持存，术语包括 rétention tertiaire 和 néguanthropie。"
            ),
            self.tr("简要填写这段录音的主题、专有名词和相关背景；留空则不启用"),
            self,
        )
        self.check_card = PushSettingCard(
            self.tr("测试连接"),
            FIF.CONNECT,
            self.tr("验证 API Key 与模型权限"),
            self.tr("仅获取上传凭证，不上传音频或提交收费转录"),
            self,
        )
        for card in (
            self.base_url_card,
            self.api_key_card,
            self.model_card,
            self.language_card,
            self.word_card,
            self.hotwords_card,
            self.vocabulary_id_card,
            self.context_card,
            self.check_card,
        ):
            self.setting_group.addSettingCard(card)
        for card in (
            self.base_url_card,
            self.api_key_card,
            self.model_card,
            self.vocabulary_id_card,
        ):
            card.lineEdit.setMinimumWidth(350)
        self.setting_group.setMinimumHeight(self.setting_group.height())
        for item in (
            cfg.qwen_filetrans_hotwords,
            cfg.qwen_filetrans_context,
            cfg.qwen_filetrans_vocabulary_id,
        ):
            item.valueChanged.connect(self._validate_options)
        self._validate_options()
        self._sync_language()
        cfg.transcribe_language.valueChanged.connect(self._sync_language)
        self.language_card.comboBox.currentIndexChanged.connect(self._on_language_changed)
        self.check_card.clicked.connect(self.check_connection)
        layout.addWidget(self.setting_group)

    def _validate_options(self, *_):
        context = cfg.qwen_filetrans_context.value.strip()
        self.context_card.validationLabel.setText(f"{len(context)} / 400 字符")
        self.context_card.validationLabel.setStyleSheet(
            "color: #c42b1c;" if len(context) > 400 else ""
        )
        try:
            vocabulary, _, _ = validate_recognition_options(cfg.qwen_filetrans_hotwords.value)
            self.hotwords_card.validationLabel.setText(
                f"{len(vocabulary)} / 2000 条；超级热词 {sum(weight == 50 for weight in vocabulary.values())} / 50 条"
            )
            self.hotwords_card.validationLabel.setStyleSheet("")
        except ValueError as error:
            self.hotwords_card.validationLabel.setText(str(error))
            self.hotwords_card.validationLabel.setStyleSheet("color: #c42b1c;")
        if len(context) > 400:
            self.context_card.validationLabel.setText("上下文最多 400 字符，请精简后重试")
        try:
            validate_recognition_options(vocabulary_id=cfg.qwen_filetrans_vocabulary_id.value)
            self.vocabulary_id_card.contentLabel.setText(
                "填入预先创建的词表 ID；与上方热词合并使用"
            )
            self.vocabulary_id_card.contentLabel.setStyleSheet("")
        except ValueError as error:
            self.vocabulary_id_card.contentLabel.setText(str(error))
            self.vocabulary_id_card.contentLabel.setStyleSheet("color: #c42b1c;")

    def validate_options(self):
        validate_recognition_options(
            cfg.qwen_filetrans_hotwords.value,
            cfg.qwen_filetrans_vocabulary_id.value,
            cfg.qwen_filetrans_context.value,
        )

    def _sync_language(self, *_):
        combo = self.language_card.comboBox
        index = combo.findData(cfg.transcribe_language.value)
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def activate(self):
        if self.language_card.comboBox.findData(cfg.transcribe_language.value) < 0:
            cfg.set(cfg.transcribe_language, TranscribeLanguageEnum.AUTO)
        self._sync_language()

    def _on_language_changed(self, index):
        cfg.set(cfg.transcribe_language, self.language_card.comboBox.itemData(index))

    def check_connection(self):
        if self.worker is not None:
            return
        self.check_card.button.setEnabled(False)
        worker = QwenFileTransConnectionThread(
            self.base_url_card.lineEdit.text(),
            self.api_key_card.lineEdit.text(),
            self.model_card.lineEdit.text(),
        )
        self.worker = worker
        worker.result_ready.connect(self._on_result)
        worker.finished.connect(self._on_finished)
        worker.start()

    def _on_result(self, success, message):
        method = InfoBar.success if success else InfoBar.error
        method(
            self.tr("连接成功" if success else "连接失败"),
            message,
            duration=5000,
            parent=self.window(),
        )

    def _on_finished(self):
        self.worker = None
        self.check_card.button.setEnabled(True)

    def stop(self):
        if self.worker is not None:
            try:
                self.worker.result_ready.disconnect(self._on_result)
                self.worker.finished.disconnect(self._on_finished)
            except (TypeError, RuntimeError):
                pass
            self.worker.stop()
            self.worker = None

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
