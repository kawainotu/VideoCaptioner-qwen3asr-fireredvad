"""Reusable MiMo pacing controls for both settings entry points."""

from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import SettingCardGroup

from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.SpinBoxSettingCard import SpinBoxSettingCard


class MiMoRateSettingWidget(SettingCardGroup):
    def __init__(self, parent=None):
        super().__init__("MiMo 请求流控", parent)
        self.rpm_card = SpinBoxSettingCard(
            cfg.mimo_rpm,
            FIF.SPEED_HIGH,
            self.tr("每分钟请求预算 (RPM)"),
            self.tr("仅按 RPM 控制，默认 80；不按 TPM 或音频时长等待，同服务地址与模型共用预算"),
            minimum=1,
            maximum=100000,
            parent=self,
        )
        self.addSettingCard(self.rpm_card)
