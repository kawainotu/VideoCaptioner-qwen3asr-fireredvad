from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    ComboBoxSettingCard,
    HyperlinkButton,
    InfoBar,
    InfoBarPosition,
    MessageBoxBase,
    ProgressBar,
    PushSettingCard,
    SegmentedWidget,
    SettingCard,
    SettingCardGroup,
    SubtitleLabel,
    SwitchSettingCard,
    TableItemDelegate,
    TableWidget,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.config import QWEN3_ASR_MODEL_PATH
from videocaptioner.core.asr.mimo_vad import check_vad_environment
from videocaptioner.core.asr.qwen3_runtime import (
    is_model_ready,
)
from videocaptioner.core.asr.qwen3_vad_models import (
    QWEN3_VAD_MODELS,
    Qwen3VADModel,
    get_qwen3_vad_model,
    is_firered_vad_model_ready,
)
from videocaptioner.core.mimo_vad_defaults import MIMO_FIRERED_VAD_DEFAULTS
from videocaptioner.core.utils.platform_utils import open_folder
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.SpinBoxSettingCard import (
    DoubleSpinBoxSettingCard,
    SpinBoxSettingCard,
)
from videocaptioner.ui.thread.huggingface_download_thread import HuggingFaceDownloadThread
from videocaptioner.ui.thread.modelscope_download_thread import ModelscopeDownloadThread
from videocaptioner.ui.thread.qwen_runtime_install_thread import QwenRuntimeInstallThread

_ACTIVE_VAD_WORKERS = set()


def reset_mimo_firered_defaults():
    for name, value in MIMO_FIRERED_VAD_DEFAULTS.items():
        cfg.set(getattr(cfg, "mimo_firered_vad_" + name), value)


def mimo_vad_components(vad_model):
    components = [
        {
            "name": "本地 VAD 共享运行环境",
            "size": "所有 VAD 共用",
            "kind": "runtime",
            "model_id": None,
            "path": None,
            "source": None,
            "ignore_patterns": (),
        }
    ]
    if vad_model.model_id:
        components.append(
            {
                "name": vad_model.label,
                "size": vad_model.size,
                "kind": "firered-vad",
                "model_id": vad_model.model_id,
                "path": vad_model.path,
                "source": vad_model.source,
                "ignore_patterns": vad_model.ignore_patterns,
            }
        )
    return tuple(components)


class MiMoVADManagerDialog(MessageBoxBase):
    def __init__(
        self,
        vad_model: Qwen3VADModel,
        parent=None,
        setting_widget=None,
    ):
        super().__init__(parent)
        self.widget.setMinimumWidth(680)
        self.vad_model = vad_model
        self.components = mimo_vad_components(vad_model)
        self.setting_widget = setting_widget
        self.active_thread = None
        self.operation_failed = False
        self._setup_ui()
        self.rejected.connect(self._on_rejected)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.addWidget(SubtitleLabel(self.tr("MiMo VAD 组件管理"), self))
        title_row.addStretch()
        open_folder_button = HyperlinkButton("", self.tr("打开模型文件夹"), self)
        open_folder_button.setIcon(FIF.FOLDER)
        open_folder_button.clicked.connect(lambda: open_folder(str(QWEN3_ASR_MODEL_PATH.parent)))
        title_row.addWidget(open_folder_button)
        layout.addLayout(title_row)
        layout.addWidget(BodyLabel(self.tr("安装当前所选 VAD 的运行环境及模型即可"), self))

        vad_selector_row = QHBoxLayout()
        vad_selector_row.addWidget(BodyLabel(self.tr("VAD 模型"), self))
        vad_selector_row.addStretch()
        self.vad_selector = SegmentedWidget(self)
        self.vad_selector.setMinimumWidth(360)
        for model in QWEN3_VAD_MODELS:
            self.vad_selector.addItem(model.key, self.tr(model.label))
        self.vad_selector.setCurrentItem(self.vad_model.key)
        self.vad_selector.currentItemChanged.connect(self._on_vad_model_changed)
        vad_selector_row.addWidget(self.vad_selector)
        layout.addLayout(vad_selector_row)

        self.table = TableWidget(self)
        self.table.setEditTriggers(TableWidget.NoEditTriggers)
        self.table.setSelectionMode(TableWidget.NoSelection)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [self.tr("组件"), self.tr("大小"), self.tr("状态"), self.tr("操作")]
        )
        self.table.setBorderVisible(True)
        self.table.setBorderRadius(8)
        self.table.setItemDelegate(TableItemDelegate(self.table))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 90)
        self.table.setColumnWidth(3, 135)
        self.table.verticalHeader().setDefaultSectionSize(48)
        self.table.setFixedHeight(178)
        layout.addWidget(self.table)

        self.progress_bar = ProgressBar(self)
        self.progress_label = BodyLabel("", self)
        self.progress_bar.hide()
        self.progress_label.hide()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

        self.viewLayout.addLayout(layout)
        self.cancelButton.setText(self.tr("关闭"))
        self.yesButton.hide()
        self._refresh_table()

    def _component_ready(self, row: int) -> bool:
        component = self.components[row]
        if component["kind"] == "runtime":
            return check_vad_environment(vad_model=self.vad_model.key, check_model=False)[0]
        if component["kind"] == "firered-vad":
            return is_firered_vad_model_ready()
        return is_model_ready(component["path"])

    def _on_vad_model_changed(self, key: str) -> None:
        model = get_qwen3_vad_model(key)
        if model.key == self.vad_model.key:
            return
        self.vad_model = model
        self.components = mimo_vad_components(model)
        if cfg.mimo_vad_model.value != model.key:
            cfg.set(cfg.mimo_vad_model, model.key)
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.components))
        self.table.setFixedHeight(34 + 48 * len(self.components))
        for row, component in enumerate(self.components):
            ready = self._component_ready(row)
            name_item = QTableWidgetItem(component["name"])
            size_item = QTableWidgetItem(component["size"])
            status_item = QTableWidgetItem(self.tr("已就绪") if ready else self.tr("未安装"))
            for item in (name_item, size_item, status_item):
                item.setTextAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
            if ready:
                status_item.setForeground(Qt.green)  # type: ignore[arg-type]
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, size_item)
            self.table.setItem(row, 2, status_item)

            container = QWidget(self)
            button_layout = QHBoxLayout(container)
            button_layout.setContentsMargins(4, 4, 4, 4)
            button = HyperlinkButton(
                "",
                self.tr("重新安装") if ready else self.tr("安装"),
                container,
            )
            button.setIcon(FIF.DOWNLOAD)
            button.setEnabled(self.active_thread is None)
            button.clicked.connect(lambda _checked=False, r=row: self._install_component(r))
            button_layout.addStretch()
            button_layout.addWidget(button)
            button_layout.addStretch()
            self.table.setCellWidget(row, 3, container)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.progress_bar.setVisible(busy)
        self.progress_label.setVisible(busy)
        if message:
            self.progress_label.setText(message)
        self._refresh_table()

    def _install_component(self, row: int) -> None:
        if self.active_thread is not None:
            return
        self.operation_failed = False
        component = self.components[row]
        self._set_busy(True, self.tr("正在准备安装"))
        if component["path"] is None:
            thread = QwenRuntimeInstallThread()
        elif component["source"] == "huggingface":
            thread = HuggingFaceDownloadThread(
                component["model_id"],
                str(component["path"]),
                component["ignore_patterns"],
            )
        else:
            thread = ModelscopeDownloadThread(component["model_id"], str(component["path"]))
        self.active_thread = thread
        _ACTIVE_VAD_WORKERS.add(thread)
        thread.finished.connect(lambda: _ACTIVE_VAD_WORKERS.discard(thread))
        thread.progress.connect(self._on_progress)
        thread.error.connect(self._on_error)
        thread.finished.connect(self._on_finished)
        thread.start()

    def _on_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_error(self, message: str) -> None:
        self.operation_failed = True
        InfoBar.error(
            self.tr("安装失败"),
            message,
            parent=self,
            duration=8000,
            position=InfoBarPosition.BOTTOM,
        )

    def _on_finished(self) -> None:
        failed = self.operation_failed
        self.active_thread = None
        self._set_busy(False)
        if not failed:
            InfoBar.success(
                self.tr("安装完成"),
                self.tr("MiMo VAD 组件已就绪"),
                parent=self,
                duration=3000,
                position=InfoBarPosition.BOTTOM,
            )
        if self.setting_widget is not None:
            self.setting_widget.refresh_status()

    def _on_rejected(self) -> None:
        if isinstance(self.active_thread, QwenRuntimeInstallThread):
            self.active_thread.stop()


class MiMoVADSettingWidget(SettingCardGroup):
    """Independent MiMo configuration, shared VAD assets and runtime."""

    def __init__(self, parent=None):
        super().__init__("MiMo VAD 设置", parent)
        self.vad_group = self
        self.vad_filter_card = SwitchSettingCard(
            FIF.CHECKBOX,
            self.tr("VAD 过滤"),
            self.tr("过滤无人声片段，减少幻觉"),
            cfg.mimo_vad_filter,
            self.vad_group,
        )
        self.vad_model_card = ComboBoxSettingCard(
            cfg.mimo_vad_model,
            FIF.ROBOT,
            self.tr("VAD 模型"),
            self.tr("选择语音活动检测模型"),
            [self.tr(model.label) for model in QWEN3_VAD_MODELS],
            self.vad_group,
        )
        self.vad_threshold_card = DoubleSpinBoxSettingCard(
            cfg.mimo_vad_threshold,
            FIF.VOLUME,  # type: ignore[arg-type]
            self.tr("Silero 语音阈值"),
            self.tr("高于此语音概率的片段会被保留"),
            minimum=0.0,
            maximum=1.0,
            decimals=2,
            step=0.05,
            parent=self.vad_group,
        )
        self.vad_min_speech_card = SpinBoxSettingCard(
            cfg.mimo_vad_min_speech_ms,
            FIF.MICROPHONE,  # type: ignore[arg-type]
            self.tr("最短语音"),
            self.tr("低于该时长的语音片段会被过滤"),
            minimum=50,
            maximum=5000,
            parent=self.vad_group,
        )
        self.vad_min_silence_card = SpinBoxSettingCard(
            cfg.mimo_vad_min_silence_ms,
            FIF.PAUSE,  # type: ignore[arg-type]
            self.tr("最短静音"),
            self.tr("达到该时长的静音会分隔语音片段"),
            minimum=50,
            maximum=5000,
            parent=self.vad_group,
        )
        self.vad_pad_card = SpinBoxSettingCard(
            cfg.mimo_vad_speech_pad_ms,
            FIF.CUT,  # type: ignore[arg-type]
            self.tr("语音边缘填充"),
            self.tr("在每段语音前后保留的毫秒数"),
            minimum=0,
            maximum=2000,
            parent=self.vad_group,
        )
        self.vad_min_speech_card.spinBox.setSuffix(" ms")
        self.vad_min_silence_card.spinBox.setSuffix(" ms")
        self.vad_pad_card.spinBox.setSuffix(" ms")

        self.firered_smooth_window_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_smooth_window_size,
            FIF.UNIT,  # type: ignore[arg-type]
            self.tr("平滑窗口"),
            self.tr("对语音概率进行平滑的连续帧数"),
            minimum=1,
            maximum=101,
            parent=self.vad_group,
        )
        self.firered_threshold_card = DoubleSpinBoxSettingCard(
            cfg.mimo_firered_vad_speech_threshold,
            FIF.VOLUME,  # type: ignore[arg-type]
            self.tr("FireRed 语音阈值"),
            self.tr("高于此概率的帧会判定为语音"),
            minimum=0.0,
            maximum=1.0,
            decimals=2,
            step=0.05,
            parent=self.vad_group,
        )
        self.firered_min_speech_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_min_speech_frame,
            FIF.MICROPHONE,  # type: ignore[arg-type]
            self.tr("最短语音帧"),
            self.tr("低于此长度的语音会被过滤，1 帧约 10 ms"),
            minimum=1,
            maximum=5000,
            parent=self.vad_group,
        )
        self.firered_max_speech_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_max_speech_frame,
            FIF.MICROPHONE,  # type: ignore[arg-type]
            self.tr("最长语音帧"),
            self.tr("达到此长度时强制切分语音，1 帧约 10 ms"),
            minimum=1,
            maximum=30000,
            parent=self.vad_group,
        )
        self.firered_min_silence_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_min_silence_frame,
            FIF.PAUSE,  # type: ignore[arg-type]
            self.tr("最短静音帧"),
            self.tr("达到此长度的静音会结束当前语音段"),
            minimum=1,
            maximum=5000,
            parent=self.vad_group,
        )
        self.firered_merge_silence_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_merge_silence_frame,
            FIF.PAUSE,  # type: ignore[arg-type]
            self.tr("合并静音帧"),
            self.tr("间隔不超过此长度的相邻语音段会被合并"),
            minimum=0,
            maximum=5000,
            parent=self.vad_group,
        )
        self.firered_extend_speech_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_extend_speech_frame,
            FIF.CUT,  # type: ignore[arg-type]
            self.tr("语音边界扩展"),
            self.tr("在检测到的语音段前后保留额外帧"),
            minimum=0,
            maximum=1000,
            parent=self.vad_group,
        )
        self.firered_chunk_max_card = SpinBoxSettingCard(
            cfg.mimo_firered_vad_chunk_max_frame,
            FIF.CUT,  # type: ignore[arg-type]
            self.tr("单次检测最大帧数"),
            self.tr("超长音频按此长度分块检测，1 帧约 10 ms"),
            minimum=100,
            maximum=60000,
            parent=self.vad_group,
        )
        for card in (
            self.firered_smooth_window_card,
            self.firered_min_speech_card,
            self.firered_max_speech_card,
            self.firered_min_silence_card,
            self.firered_merge_silence_card,
            self.firered_extend_speech_card,
            self.firered_chunk_max_card,
        ):
            card.spinBox.setSuffix(" 帧")

        for card in (
            self.vad_filter_card,
            self.vad_model_card,
            self.vad_threshold_card,
            self.vad_min_speech_card,
            self.vad_min_silence_card,
            self.vad_pad_card,
            self.firered_smooth_window_card,
            self.firered_threshold_card,
            self.firered_min_speech_card,
            self.firered_max_speech_card,
            self.firered_min_silence_card,
            self.firered_merge_silence_card,
            self.firered_extend_speech_card,
            self.firered_chunk_max_card,
        ):
            self.vad_group.addSettingCard(card)
        self.reset_card = PushSettingCard(
            self.tr("恢复默认参数"),
            FIF.SYNC,
            self.tr("FireRed 标准参数"),
            self.tr("采用官方非流式基线，可按音频调整；仅重置 MiMo 的八项 FireRed 参数"),
            self,
        )
        self.addSettingCard(self.reset_card)
        self.reset_card.clicked.connect(reset_mimo_firered_defaults)
        self.manage_card = SettingCard(FIF.DOWNLOAD, self.tr("VAD 组件"), "", self.vad_group)
        self.manage_button = HyperlinkButton("", self.tr("准备环境 / 下载模型"), self.manage_card)
        self.manage_card.hBoxLayout.addWidget(self.manage_button)
        self.vad_group.addSettingCard(self.manage_card)
        self.manage_button.clicked.connect(self.open_manager)
        cfg.mimo_vad_filter.valueChanged.connect(self.refresh_status)
        cfg.mimo_vad_model.valueChanged.connect(self.refresh_status)
        self.refresh_status()

    def refresh_status(self, *args):
        enabled = cfg.mimo_vad_filter.value
        firered = cfg.mimo_vad_model.value == "firered"
        self.vad_model_card.setEnabled(enabled)
        for card in (
            self.vad_threshold_card,
            self.vad_min_speech_card,
            self.vad_min_silence_card,
            self.vad_pad_card,
        ):
            card.setVisible(enabled and not firered)
        for card in (
            self.firered_smooth_window_card,
            self.firered_threshold_card,
            self.firered_min_speech_card,
            self.firered_max_speech_card,
            self.firered_min_silence_card,
            self.firered_merge_silence_card,
            self.firered_extend_speech_card,
            self.firered_chunk_max_card,
        ):
            card.setVisible(enabled and firered)
        self.reset_card.setVisible(enabled and firered)
        self.manage_card.setVisible(enabled)
        if enabled:
            ready, message = check_vad_environment(vad_model=cfg.mimo_vad_model.value)
            self.manage_card.setContent(message)
        height = self.cardLayout.heightForWidth(self.width()) + 46
        self.setFixedHeight(height)
        if isinstance(self.parentWidget(), SettingCardGroup):
            self.parentWidget().adjustSize()

    def open_manager(self):
        existing = getattr(self, "_manager", None)
        if existing is not None and existing.active_thread is not None:
            existing.exec_()
            return
        dialog = MiMoVADManagerDialog(
            get_qwen3_vad_model(cfg.mimo_vad_model.value), self.window(), setting_widget=self
        )
        # Keep the dialog and its active worker alive if the user closes it mid-download.
        self._manager = dialog
        dialog.exec_()
