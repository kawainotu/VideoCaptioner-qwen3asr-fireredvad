# -*- coding: utf-8 -*-
from typing import Optional, Set

from openai import OpenAI
from PyQt5.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget
from qfluentwidgets import (
    ComboBoxSettingCard,
    InfoBar,
    InfoBarPosition,
    PushSettingCard,
    SettingCardGroup,
    SingleDirectionScrollArea,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.core.constant import (
    INFOBAR_DURATION_ERROR,
    INFOBAR_DURATION_SUCCESS,
)
from videocaptioner.core.entities import MiMoLanguageEnum
from videocaptioner.core.llm.check_mimo import (
    _sanitize_error,
    check_mimo_connection,
    validate_and_normalize_mimo_base_url,
)
from videocaptioner.ui.components.MiMoRateSettingWidget import MiMoRateSettingWidget
from videocaptioner.ui.components.MiMoVADSettingWidget import MiMoVADSettingWidget
from videocaptioner.ui.components.MiMoAlignmentSettingWidget import MiMoAlignmentSettingWidget

from ..common.config import cfg
from .LineEditSettingCard import LineEditSettingCard

# 模块级全局活跃 Worker 注册表，保障即使 Widget 先行销毁，线程所有权依然存续至安全退出
_ACTIVE_MIMO_WORKERS: Set["MiMoConnectionThread"] = set()
_LAST_CONNECTED_APP = None


def _cleanup_all_mimo_workers():
    """在应用准备退出时停止所有活跃 Worker。"""
    for worker in list(_ACTIVE_MIMO_WORKERS):
        worker.stop()


def _ensure_about_to_quit_connected():
    """动态检查并确保 QCoreApplication 的 aboutToQuit 信号已连接到全局停止函数。"""
    global _LAST_CONNECTED_APP
    app = QCoreApplication.instance()
    if app is not None and app is not _LAST_CONNECTED_APP:
        try:
            app.aboutToQuit.connect(_cleanup_all_mimo_workers)
            _LAST_CONNECTED_APP = app
        except Exception:
            pass


class MiMoConnectionThread(QThread):
    """MiMo API 连通性异步测试线程。

    具备独立的结果信号、严格的 HTTP 超时 (10s) 与客户端关闭能力，
    不覆盖 Qt 原生 finished 信号，对所有异常进行凭据脱敏。
    自动接入全局 Worker 注册表，并在真实 finished 时执行模块级清理与 deleteLater。
    """

    result_ready = pyqtSignal(bool, str)

    def __init__(self, base_url: str, api_key: str, model: str, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._is_stopped = False
        self._client: Optional[OpenAI] = None

        # 确保应用退出信号已连接，并将自身加入全局活跃集合
        _ensure_about_to_quit_connected()
        _ACTIVE_MIMO_WORKERS.add(self)
        self.finished.connect(self._on_finished_cleanup)

    def _on_finished_cleanup(self) -> None:
        """模块级生命周期统一回收：从全局集合移除并安全析构。"""
        _ACTIVE_MIMO_WORKERS.discard(self)
        self.deleteLater()

    def stop(self) -> None:
        """安全停止/取消测试，并关闭底层连接。"""
        self._is_stopped = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    def run(self) -> None:
        if self._is_stopped:
            return

        try:
            # 建立受限超时的 OpenAI 客户端 (timeout=10, max_retries=0)
            norm_url = validate_and_normalize_mimo_base_url(self.base_url)
            self._client = OpenAI(
                base_url=norm_url,
                api_key=self.api_key,
                timeout=10,
                max_retries=0,
            )

            success, message = check_mimo_connection(
                self.base_url,
                self.api_key,
                self.model,
                client=self._client,
                rpm=cfg.mimo_rpm.value, tpm=cfg.mimo_tpm.value,
                cancelled=lambda: self._is_stopped,
            )
            if not self._is_stopped:
                self.result_ready.emit(success, message)
        except Exception as e:
            if not self._is_stopped:
                clean_err = _sanitize_error(str(e), self.api_key)
                self.result_ready.emit(False, f"测试异常: {clean_err}")
        finally:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None


class MiMoASRSettingWidget(QWidget):
    """Xiaomi MiMo-ASR dedicated settings widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_worker: Optional[MiMoConnectionThread] = None
        self.setup_ui()
        # 当宿主组件销毁时，仅停止自身发起的 Worker，避免影响其他无关 Worker
        self.destroyed.connect(self._stop_own_worker)

    def _stop_own_worker(self):
        """仅停止当前组件发起的 Worker 并断开结果信号，避免访问已销毁的 QWidget。"""
        if self._current_worker is not None:
            try:
                self._current_worker.result_ready.disconnect(
                    self.on_connection_check_result
                )
            except Exception:
                pass
            self._current_worker.stop()
            self._current_worker = None

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        # 创建单向滚动区域和容器
        self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)  # type: ignore
        self.scrollArea.setStyleSheet(
            "QScrollArea{background: transparent; border: none}"
        )

        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.containerLayout = QVBoxLayout(self.container)

        self.setting_group = SettingCardGroup(self.tr("MiMo-ASR [API] 设置"), self)

        # API Base URL (保留用户自定义路径，提供中立有效示例占位符)
        self.base_url_card = LineEditSettingCard(
            cfg.mimo_api_base,
            FIF.LINK,
            self.tr("API Base URL"),
            self.tr("小米 MiMo API 基础地址（支持自定义与 Token Plan 渠道，必填）"),
            self.tr("请填写对应渠道的 Base URL (如 https://api.xiaomimimo.com/v1)"),
            self.setting_group,
        )

        # API Key (密码回显模式)
        self.api_key_card = LineEditSettingCard(
            cfg.mimo_api_key,
            FIF.FINGERPRINT,
            self.tr("API Key"),
            self.tr("输入小米 MiMo API Key（必填）"),
            "sk-",
            self.setting_group,
        )
        self.api_key_card.lineEdit.setEchoMode(QLineEdit.EchoMode.Password)

        # Model
        self.model_card = LineEditSettingCard(
            cfg.mimo_api_model,
            FIF.ROBOT,  # type: ignore
            self.tr("模型名称"),
            self.tr("识别模型名称（默认 mimo-v2.5-asr）"),
            "mimo-v2.5-asr",
            self.setting_group,
        )

        # 语言选择 (使用独立持久化的 mimo_transcribe_language，完全隔离全局语言配置)
        self.language_card = ComboBoxSettingCard(
            cfg.mimo_transcribe_language,
            FIF.LANGUAGE,
            self.tr("源语言"),
            self.tr("音频中说话的语言（支持自动检测、中文、英语）"),
            [
                MiMoLanguageEnum.AUTO.value,
                MiMoLanguageEnum.CHINESE.value,
                MiMoLanguageEnum.ENGLISH.value,
            ],
            self.setting_group,
        )

        # 测试连接按钮
        self.check_connection_card = PushSettingCard(
            self.tr("测试连接"),
            FIF.CONNECT,
            self.tr("测试 MiMo API 连接"),
            self.tr("使用当前表单填写的值向语音识别端点发送微小探针，测试 API 连通性"),
            self.setting_group,
        )

        # 控件宽度设置
        self.base_url_card.lineEdit.setMinimumWidth(260)
        self.api_key_card.lineEdit.setMinimumWidth(260)
        self.model_card.lineEdit.setMinimumWidth(260)
        self.language_card.comboBox.setMinimumWidth(260)

        # 添加卡片到组
        self.setting_group.addSettingCard(self.base_url_card)
        self.setting_group.addSettingCard(self.api_key_card)
        self.setting_group.addSettingCard(self.model_card)
        self.setting_group.addSettingCard(self.language_card)
        self.setting_group.addSettingCard(self.check_connection_card)

        # 信号连接
        self.check_connection_card.clicked.connect(self.on_check_connection)

        self.vad_widget = MiMoVADSettingWidget(self)
        self.vad_filter_card = self.vad_widget.vad_filter_card
        self.containerLayout.addWidget(self.setting_group)
        self.containerLayout.addWidget(self.vad_widget)
        self.alignment_widget = MiMoAlignmentSettingWidget(self)
        self.containerLayout.addWidget(self.alignment_widget)
        self.rate_widget = MiMoRateSettingWidget(self)
        self.containerLayout.addWidget(self.rate_widget)
        self.containerLayout.addStretch(1)

        self.scrollArea.setWidget(self.container)
        self.scrollArea.setWidgetResizable(True)
        self.main_layout.addWidget(self.scrollArea)

    def on_check_connection(self):
        """测试 MiMo API 连接（使用当前未保存的表单值，具备全局防连点保护）"""
        # 防止并发测试
        if self._current_worker is not None and self._current_worker.isRunning():
            return
        if any(w.isRunning() for w in _ACTIVE_MIMO_WORKERS):
            return

        base_url = self.base_url_card.lineEdit.text().strip()
        api_key = self.api_key_card.lineEdit.text().strip()
        model = self.model_card.lineEdit.text().strip() or "mimo-v2.5-asr"

        if not base_url:
            InfoBar.warning(
                self.tr("配置不完整"),
                self.tr("请输入 MiMo API Base URL"),
                duration=INFOBAR_DURATION_ERROR,
                position=InfoBarPosition.TOP,
                parent=self.window(),
            )
            return

        if not api_key:
            InfoBar.warning(
                self.tr("配置不完整"),
                self.tr("请输入 MiMo API Key"),
                duration=INFOBAR_DURATION_ERROR,
                position=InfoBarPosition.TOP,
                parent=self.window(),
            )
            return

        # 禁用按钮，显示正在测试
        self.check_connection_card.button.setEnabled(False)
        self.check_connection_card.button.setText(self.tr("正在测试..."))

        # 启动安全管理的测试线程（parent=None 确保生命周期脱钩，模块级自动托管）
        worker = MiMoConnectionThread(base_url, api_key, model, parent=None)
        self._current_worker = worker

        worker.result_ready.connect(self.on_connection_check_result)
        worker.start()

    def on_connection_check_result(self, success: bool, result: str):
        """处理连接测试完成结果"""
        self._current_worker = None
        self.check_connection_card.button.setEnabled(True)
        self.check_connection_card.button.setText(self.tr("测试连接"))

        if success:
            InfoBar.success(
                self.tr("连接成功"),
                result,
                duration=INFOBAR_DURATION_SUCCESS,
                position=InfoBarPosition.BOTTOM,
                parent=self.window(),
            )
        else:
            InfoBar.error(
                self.tr("连接失败"),
                result,
                duration=INFOBAR_DURATION_ERROR,
                position=InfoBarPosition.BOTTOM,
                parent=self.window(),
            )

    def closeEvent(self, event):
        """关闭时停止自身 worker 并断开回调。"""
        self._stop_own_worker()
        super().closeEvent(event)
