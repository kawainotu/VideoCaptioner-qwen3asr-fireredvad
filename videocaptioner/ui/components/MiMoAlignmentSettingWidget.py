"""Shared MiMo alignment controls for the transcription and settings pages."""

from qfluentwidgets import ComboBoxSettingCard, PushSettingCard, SettingCardGroup
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.config import QWEN3_ALIGNER_MODEL_PATH
from videocaptioner.core.asr.qwen3_models import get_qwen3_asr_model
from videocaptioner.core.asr.qwen3_runtime import is_model_ready, is_runtime_ready
from videocaptioner.core.asr.qwen3_vad_models import get_qwen3_vad_model
from videocaptioner.ui.common.config import cfg

from .Qwen3ASRSettingWidget import Qwen3ASRManagerDialog


class MiMoAlignmentSettingWidget(SettingCardGroup):
    def __init__(self, parent=None):
        super().__init__("MiMo 短字幕与本地时间对齐", parent)
        self._manager = None
        self.prepare_card = PushSettingCard(
            self.tr("准备 / 查看组件"), FIF.DOWNLOAD,
            self.tr("Qwen3-ForcedAligner-0.6B"), "", self,
        )
        self.device_card = ComboBoxSettingCard(
            cfg.mimo_aligner_device, FIF.SPEED_HIGH,
            self.tr("本地对齐设备"),
            self.tr("MiMo 识别后在本机对齐时间，再生成短字幕；缺少组件时将明确报错。"),
            ["auto", "cuda", "cpu"], self,
        )
        self.addSettingCard(self.prepare_card)
        self.addSettingCard(self.device_card)
        self.prepare_card.clicked.connect(self.open_manager)
        self.refresh_status()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_status()

    def refresh_status(self):
        runtime = self.tr("已就绪") if is_runtime_ready() else self.tr("未安装")
        model = self.tr("已就绪") if is_model_ready(QWEN3_ALIGNER_MODEL_PATH) else self.tr("未下载")
        self.prepare_card.setContent(
            self.tr("共享运行环境：") + runtime + self.tr("；对齐模型：") + model
            + self.tr("。仅下载对齐权重，无需 Qwen 识别模型；已有组件直接复用。")
        )
        self.setFixedHeight(self.cardLayout.heightForWidth(self.width()) + 46)
        if isinstance(self.parentWidget(), SettingCardGroup):
            self.parentWidget().adjustSize()

    def open_manager(self):
        if self._manager is not None and self._manager.active_thread is not None:
            self._manager.exec_()
            self.refresh_status()
            return
        dialog = Qwen3ASRManagerDialog(
            get_qwen3_asr_model(cfg.qwen_asr_model.value),
            get_qwen3_vad_model(cfg.qwen_asr_vad_model.value),
            parent=self.window(), setting_widget=self, aligner_only=True,
        )
        self._manager = dialog
        dialog.exec_()
        self.refresh_status()
