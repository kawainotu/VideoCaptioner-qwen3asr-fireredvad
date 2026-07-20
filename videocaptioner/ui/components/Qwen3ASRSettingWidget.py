from PyQt5.QtCore import Qt
from PyQt5.QtGui import QShowEvent
from PyQt5.QtWidgets import QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    ComboBoxSettingCard,
    HyperlinkButton,
    HyperlinkCard,
    InfoBar,
    InfoBarPosition,
    MessageBoxBase,
    ProgressBar,
    SegmentedWidget,
    SettingCard,
    SettingCardGroup,
    SingleDirectionScrollArea,
    SubtitleLabel,
    SwitchSettingCard,
    TableItemDelegate,
    TableWidget,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.config import QWEN3_ALIGNER_MODEL_PATH, QWEN3_ASR_MODEL_PATH
from videocaptioner.core.asr.qwen3_models import (
    QWEN3_ASR_MODELS,
    Qwen3ASRModel,
    get_qwen3_asr_model,
)
from videocaptioner.core.asr.qwen3_runtime import (
    is_model_ready,
    is_runtime_ready,
    missing_components,
)
from videocaptioner.core.asr.qwen3_vad_models import (
    FIRERED_VAD_MODEL_KEY,
    QWEN3_VAD_MODELS,
    Qwen3VADModel,
    get_qwen3_vad_model,
    is_firered_vad_model_ready,
)
from videocaptioner.core.entities import TranscribeLanguageEnum
from videocaptioner.core.utils.platform_utils import open_folder
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.LineEditSettingCard import LineEditSettingCard
from videocaptioner.ui.components.SpinBoxSettingCard import (
    DoubleSpinBoxSettingCard,
    SpinBoxSettingCard,
)
from videocaptioner.ui.thread.huggingface_download_thread import HuggingFaceDownloadThread
from videocaptioner.ui.thread.modelscope_download_thread import ModelscopeDownloadThread
from videocaptioner.ui.thread.qwen_runtime_install_thread import QwenRuntimeInstallThread


def qwen_components(
    asr_model: Qwen3ASRModel, vad_model: Qwen3VADModel
) -> tuple[dict, ...]:
    """Build the component list for the selected ASR checkpoint."""
    components = [
        {
            "name": "Qwen3-ASR 共享运行环境",
            "size": "所有模型共用",
            "kind": "runtime",
            "model_id": None,
            "path": None,
            "source": None,
            "ignore_patterns": (),
        },
        {
            "name": asr_model.model_id,
            "size": asr_model.size,
            "kind": "asr",
            "model_id": asr_model.model_id,
            "path": asr_model.path,
            "source": asr_model.source,
            "ignore_patterns": asr_model.ignore_patterns,
        },
        {
            "name": "Qwen3-ForcedAligner-0.6B",
            "size": "1.84 GB",
            "kind": "aligner",
            "model_id": "Qwen/Qwen3-ForcedAligner-0.6B",
            "path": QWEN3_ALIGNER_MODEL_PATH,
            "source": "modelscope",
            "ignore_patterns": (),
        },
    ]
    if vad_model.model_id and vad_model.path:
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

QWEN_TIMESTAMP_LANGUAGES = (
    TranscribeLanguageEnum.AUTO,
    TranscribeLanguageEnum.CHINESE,
    TranscribeLanguageEnum.ENGLISH,
    TranscribeLanguageEnum.YUE,
    TranscribeLanguageEnum.FRENCH,
    TranscribeLanguageEnum.GERMAN,
    TranscribeLanguageEnum.SPANISH,
    TranscribeLanguageEnum.RUSSIAN,
    TranscribeLanguageEnum.PORTUGUESE,
    TranscribeLanguageEnum.ITALIAN,
    TranscribeLanguageEnum.JAPANESE,
    TranscribeLanguageEnum.KOREAN,
)


class Qwen3ASRManagerDialog(MessageBoxBase):
    def __init__(
        self,
        asr_model: Qwen3ASRModel,
        vad_model: Qwen3VADModel,
        parent=None,
        setting_widget=None,
    ):
        super().__init__(parent)
        self.widget.setMinimumWidth(680)
        self.asr_model = asr_model
        self.vad_model = vad_model
        self.components = qwen_components(asr_model, vad_model)
        self.setting_widget = setting_widget
        self.active_thread = None
        self.operation_failed = False
        self._setup_ui()
        self.rejected.connect(self._on_rejected)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.addWidget(SubtitleLabel(self.tr("Qwen3-ASR 组件管理"), self))
        title_row.addStretch()
        open_folder_button = HyperlinkButton("", self.tr("打开模型文件夹"), self)
        open_folder_button.setIcon(FIF.FOLDER)
        open_folder_button.clicked.connect(lambda: open_folder(str(QWEN3_ASR_MODEL_PATH.parent)))
        title_row.addWidget(open_folder_button)
        layout.addLayout(title_row)
        layout.addWidget(
            BodyLabel(
                self.tr("请安装识别、强制对齐及当前所选 VAD 所需的组件"), self
            )
        )

        model_selector_row = QHBoxLayout()
        model_selector_row.addWidget(BodyLabel(self.tr("Qwen3 系列模型"), self))
        model_selector_row.addStretch()
        self.model_selector = SegmentedWidget(self)
        self.model_selector.setMinimumWidth(360)
        for model in QWEN3_ASR_MODELS:
            self.model_selector.addItem(model.key, self.tr(model.label))
        self.model_selector.setCurrentItem(self.asr_model.key)
        self.model_selector.currentItemChanged.connect(self._on_model_changed)
        model_selector_row.addWidget(self.model_selector)
        layout.addLayout(model_selector_row)

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
            return is_runtime_ready()
        if component["kind"] == "firered-vad":
            return is_firered_vad_model_ready()
        return is_model_ready(component["path"])

    def _on_model_changed(self, key: str) -> None:
        model = get_qwen3_asr_model(key)
        if model.key == self.asr_model.key:
            return
        self.asr_model = model
        self.components = qwen_components(model, self.vad_model)
        if cfg.qwen_asr_model.value != model.key:
            cfg.set(cfg.qwen_asr_model, model.key)
        self._refresh_table()

    def _on_vad_model_changed(self, key: str) -> None:
        model = get_qwen3_vad_model(key)
        if model.key == self.vad_model.key:
            return
        self.vad_model = model
        self.components = qwen_components(self.asr_model, model)
        if cfg.qwen_asr_vad_model.value != model.key:
            cfg.set(cfg.qwen_asr_vad_model, model.key)
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.components))
        self.table.setFixedHeight(34 + 48 * len(self.components))
        for row, component in enumerate(self.components):
            ready = self._component_ready(row)
            name_item = QTableWidgetItem(component["name"])
            size_item = QTableWidgetItem(component["size"])
            status_item = QTableWidgetItem(
                self.tr("已就绪") if ready else self.tr("未安装")
            )
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
                self.tr("Qwen3-ASR 组件已就绪"),
                parent=self,
                duration=3000,
                position=InfoBarPosition.BOTTOM,
            )
        if self.setting_widget is not None:
            self.setting_widget.refresh_status()

    def _on_rejected(self) -> None:
        if isinstance(self.active_thread, QwenRuntimeInstallThread):
            self.active_thread.stop()


class Qwen3ASRSettingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._missing_prompted = False
        self._setup_ui()
        self._connect_signals()
        self.refresh_status()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        asr_model = get_qwen3_asr_model(cfg.qwen_asr_model.value)
        if not self._missing_prompted and missing_components(
            asr_model_dir=asr_model.path,
            vad_model_key=(
                cfg.qwen_asr_vad_model.value if cfg.qwen_asr_vad_filter.value else None
            ),
        ):
            self._missing_prompted = True
            InfoBar.warning(
                self.tr("Qwen3-ASR 尚未就绪"),
                self.tr("请先安装运行环境、识别模型、时间戳模型和所选 VAD"),
                parent=self.window(),
                duration=5000,
                position=InfoBarPosition.BOTTOM,
            )

    def _setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.scroll_area = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)  # type: ignore[arg-type]
        self.scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")
        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.container_layout = QVBoxLayout(self.container)

        self.model_group = SettingCardGroup(self.tr("Qwen3-ASR 设置"), self)
        self.model_card = SettingCard(
            FIF.ROBOT,
            self.tr("模型"),
            self.tr("选择识别模型并安装 Qwen3-ForcedAligner-0.6B"),
            self.model_group,
        )
        self.model_selector_card = SettingCard(
            FIF.ROBOT,
            self.tr("Qwen3 系列模型"),
            self.tr("选择用于语音识别的模型"),
            self.model_group,
        )
        self.model_selector = SegmentedWidget(self.model_selector_card)
        self.model_selector_card.hBoxLayout.addWidget(
            self.model_selector, 0, Qt.AlignRight  # type: ignore[arg-type]
        )
        self.model_selector_card.hBoxLayout.addSpacing(16)
        for model in QWEN3_ASR_MODELS:
            self.model_selector.addItem(model.key, self.tr(model.label))
        self._set_qwen_model(cfg.qwen_asr_model.value)
        self.manage_card = HyperlinkCard(
            "",
            self.tr("管理组件"),
            FIF.DOWNLOAD,
            self.tr("运行环境与模型"),
            self.tr("安装或更新 Qwen3-ASR、时间戳与 VAD 组件"),
            self.model_group,
        )
        self.device_card = ComboBoxSettingCard(
            cfg.qwen_asr_device,
            FIF.IOT,
            self.tr("运行设备"),
            self.tr("模型运行设备"),
            ["auto", "cuda", "cpu"],
            self.model_group,
        )
        self.low_memory_card = SwitchSettingCard(
            FIF.SAVE,
            self.tr("低显存模式"),
            self.tr("识别与时间戳对齐分阶段加载模型"),
            cfg.qwen_asr_low_memory,
            self.model_group,
        )

        self._qwen_languages = QWEN_TIMESTAMP_LANGUAGES
        if cfg.transcribe_language.value not in self._qwen_languages:
            cfg.set(cfg.transcribe_language, TranscribeLanguageEnum.AUTO)
        # ComboBoxSettingCard pairs its labels with every option from the
        # config item. Qwen exposes only a subset, so bind this selector
        # explicitly to avoid labels and persisted enum values drifting apart.
        self.language_card = SettingCard(
            FIF.LANGUAGE,
            self.tr("源语言"),
            self.tr("自动检测或选择支持精确时间戳的语言"),
            self.model_group,
        )
        self.language_card.comboBox = ComboBox(self.language_card)
        self.language_card.hBoxLayout.addWidget(
            self.language_card.comboBox, 0, Qt.AlignRight  # type: ignore[arg-type]
        )
        self.language_card.hBoxLayout.addSpacing(16)
        for language in self._qwen_languages:
            self.language_card.comboBox.addItem(language.value, userData=language)
        self._set_qwen_language(cfg.transcribe_language.value)

        self.vad_group = SettingCardGroup(self.tr("VAD 设置"), self)
        self.vad_filter_card = SwitchSettingCard(
            FIF.CHECKBOX,
            self.tr("VAD 过滤"),
            self.tr("过滤无人声片段，减少幻觉"),
            cfg.qwen_asr_vad_filter,
            self.vad_group,
        )
        self.vad_model_card = ComboBoxSettingCard(
            cfg.qwen_asr_vad_model,
            FIF.ROBOT,
            self.tr("VAD 模型"),
            self.tr("选择语音活动检测模型"),
            [self.tr(model.label) for model in QWEN3_VAD_MODELS],
            self.vad_group,
        )
        self.vad_threshold_card = DoubleSpinBoxSettingCard(
            cfg.qwen_asr_vad_threshold,
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
            cfg.qwen_asr_vad_min_speech_ms,
            FIF.MICROPHONE,  # type: ignore[arg-type]
            self.tr("最短语音"),
            self.tr("低于该时长的语音片段会被过滤"),
            minimum=50,
            maximum=5000,
            parent=self.vad_group,
        )
        self.vad_min_silence_card = SpinBoxSettingCard(
            cfg.qwen_asr_vad_min_silence_ms,
            FIF.PAUSE,  # type: ignore[arg-type]
            self.tr("最短静音"),
            self.tr("达到该时长的静音会分隔语音片段"),
            minimum=50,
            maximum=5000,
            parent=self.vad_group,
        )
        self.vad_pad_card = SpinBoxSettingCard(
            cfg.qwen_asr_vad_speech_pad_ms,
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
            cfg.qwen_asr_firered_vad_smooth_window_size,
            FIF.UNIT,  # type: ignore[arg-type]
            self.tr("平滑窗口"),
            self.tr("对语音概率进行平滑的连续帧数"),
            minimum=1,
            maximum=101,
            parent=self.vad_group,
        )
        self.firered_threshold_card = DoubleSpinBoxSettingCard(
            cfg.qwen_asr_firered_vad_speech_threshold,
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
            cfg.qwen_asr_firered_vad_min_speech_frame,
            FIF.MICROPHONE,  # type: ignore[arg-type]
            self.tr("最短语音帧"),
            self.tr("低于此长度的语音会被过滤，1 帧约 10 ms"),
            minimum=1,
            maximum=5000,
            parent=self.vad_group,
        )
        self.firered_max_speech_card = SpinBoxSettingCard(
            cfg.qwen_asr_firered_vad_max_speech_frame,
            FIF.MICROPHONE,  # type: ignore[arg-type]
            self.tr("最长语音帧"),
            self.tr("达到此长度时强制切分语音，1 帧约 10 ms"),
            minimum=1,
            maximum=30000,
            parent=self.vad_group,
        )
        self.firered_min_silence_card = SpinBoxSettingCard(
            cfg.qwen_asr_firered_vad_min_silence_frame,
            FIF.PAUSE,  # type: ignore[arg-type]
            self.tr("最短静音帧"),
            self.tr("达到此长度的静音会结束当前语音段"),
            minimum=1,
            maximum=5000,
            parent=self.vad_group,
        )
        self.firered_merge_silence_card = SpinBoxSettingCard(
            cfg.qwen_asr_firered_vad_merge_silence_frame,
            FIF.PAUSE,  # type: ignore[arg-type]
            self.tr("合并静音帧"),
            self.tr("间隔不超过此长度的相邻语音段会被合并"),
            minimum=0,
            maximum=5000,
            parent=self.vad_group,
        )
        self.firered_extend_speech_card = SpinBoxSettingCard(
            cfg.qwen_asr_firered_vad_extend_speech_frame,
            FIF.CUT,  # type: ignore[arg-type]
            self.tr("语音边界扩展"),
            self.tr("在检测到的语音段前后保留额外帧"),
            minimum=0,
            maximum=1000,
            parent=self.vad_group,
        )
        self.firered_chunk_max_card = SpinBoxSettingCard(
            cfg.qwen_asr_firered_vad_chunk_max_frame,
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

        self.other_group = SettingCardGroup(self.tr("其他设置"), self)
        self.timestamp_card = SwitchSettingCard(
            FIF.UNIT,
            self.tr("字词级时间戳"),
            self.tr("开启后，语音转录页面会在输入文件同目录导出字词级字幕"),
            cfg.qwen_asr_word_timestamps,
            self.other_group,
        )
        self.prompt_card = LineEditSettingCard(
            cfg.qwen_asr_prompt,
            FIF.CHAT,
            self.tr("上下文提示"),
            self.tr("可选，用于补充人名、术语或领域上下文"),
            "",
            self.other_group,
        )

        for card in (
            self.model_card,
            self.model_selector_card,
            self.manage_card,
            self.device_card,
            self.low_memory_card,
            self.language_card,
        ):
            self.model_group.addSettingCard(card)
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
        self.other_group.addSettingCard(self.timestamp_card)
        self.other_group.addSettingCard(self.prompt_card)

        self.container_layout.addWidget(self.model_group)
        self.container_layout.addWidget(self.vad_group)
        self.container_layout.addWidget(self.other_group)
        self.container_layout.addStretch(1)
        self.device_card.comboBox.setMinimumWidth(200)
        self.model_selector.setMinimumWidth(360)
        self.language_card.comboBox.setMinimumWidth(200)
        self.vad_model_card.comboBox.setMinimumWidth(200)
        self.prompt_card.lineEdit.setMinimumWidth(200)
        self.scroll_area.setWidget(self.container)
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

    def _connect_signals(self) -> None:
        self.manage_card.linkButton.clicked.connect(self._show_manager)
        self.model_selector.currentItemChanged.connect(self._on_qwen_model_changed)
        cfg.qwen_asr_model.valueChanged.connect(self._on_qwen_model_config_changed)
        self.language_card.comboBox.currentIndexChanged.connect(self._on_qwen_language_changed)
        cfg.transcribe_language.valueChanged.connect(self._set_qwen_language)
        self.vad_filter_card.checkedChanged.connect(self._on_vad_filter_changed)
        cfg.qwen_asr_vad_model.valueChanged.connect(self._on_vad_model_changed)
        self._on_vad_filter_changed(cfg.qwen_asr_vad_filter.value)

    def _on_qwen_language_changed(self, index: int) -> None:
        language = self.language_card.comboBox.itemData(index)
        if language in self._qwen_languages and language != cfg.transcribe_language.value:
            cfg.set(cfg.transcribe_language, language)

    def _on_qwen_model_changed(self, key: str) -> None:
        if key != cfg.qwen_asr_model.value:
            cfg.set(cfg.qwen_asr_model, key)

    def _on_qwen_model_config_changed(self, key: str) -> None:
        self._set_qwen_model(key)
        self.refresh_status()

    def _set_qwen_model(self, key: str) -> None:
        model = get_qwen3_asr_model(key)
        if self.model_selector.currentRouteKey() != model.key:
            self.model_selector.setCurrentItem(model.key)
        self.model_selector_card.setContent(self.tr(model.description))

    def _set_qwen_language(self, language: TranscribeLanguageEnum) -> None:
        if language not in self._qwen_languages:
            return
        index = self.language_card.comboBox.findData(language)
        if index >= 0 and index != self.language_card.comboBox.currentIndex():
            self.language_card.comboBox.setCurrentIndex(index)

    def _on_vad_filter_changed(self, checked: bool) -> None:
        self.vad_model_card.setEnabled(checked)
        self._update_vad_cards()
        self.refresh_status()

    def _on_vad_model_changed(self, _key: str) -> None:
        self._update_vad_cards()
        self.refresh_status()

    def _update_vad_cards(self) -> None:
        enabled = cfg.qwen_asr_vad_filter.value
        use_firered = (
            get_qwen3_vad_model(cfg.qwen_asr_vad_model.value).key
            == FIRERED_VAD_MODEL_KEY
        )
        silero_cards = (
            self.vad_threshold_card,
            self.vad_min_speech_card,
            self.vad_min_silence_card,
            self.vad_pad_card,
        )
        firered_cards = (
            self.firered_smooth_window_card,
            self.firered_threshold_card,
            self.firered_min_speech_card,
            self.firered_max_speech_card,
            self.firered_min_silence_card,
            self.firered_merge_silence_card,
            self.firered_extend_speech_card,
            self.firered_chunk_max_card,
        )
        for card in silero_cards:
            card.setVisible(not use_firered)
            card.setEnabled(enabled)
        for card in firered_cards:
            card.setVisible(use_firered)
            card.setEnabled(enabled)

    def _show_manager(self) -> None:
        model = get_qwen3_asr_model(cfg.qwen_asr_model.value)
        vad_model = get_qwen3_vad_model(cfg.qwen_asr_vad_model.value)
        Qwen3ASRManagerDialog(model, vad_model, self.window(), self).exec_()

    def refresh_status(self) -> None:
        model = get_qwen3_asr_model(cfg.qwen_asr_model.value)
        vad_model = get_qwen3_vad_model(cfg.qwen_asr_vad_model.value)
        missing = missing_components(
            asr_model_dir=model.path,
            vad_model_key=vad_model.key if cfg.qwen_asr_vad_filter.value else None,
        )
        if missing:
            self.model_card.setContent(
                f"{model.label}；{self.tr('缺少：')}{', '.join(missing)}"
            )
        else:
            components = f"{model.label} + Qwen3-ForcedAligner-0.6B"
            if cfg.qwen_asr_vad_filter.value:
                components += f" + {vad_model.label}"
            self.model_card.setContent(f"{components}{self.tr('（已就绪）')}")
