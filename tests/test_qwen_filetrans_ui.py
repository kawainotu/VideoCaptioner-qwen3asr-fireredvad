"""Real Qt layouts in a separate process with isolated config and no API calls."""

import os
import subprocess
import sys
from pathlib import Path


def test_full_transcription_and_home_forms_scroll_at_small_and_scaled_sizes(tmp_path):
    script = r"""
import json
import sys
from pathlib import Path
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QPalette, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QSizePolicy
app = QApplication([])
import requests
def no_network(*args, **kwargs):
    raise AssertionError('UI verification must not access the network')
requests.sessions.Session.request = no_network
for name in ('msyh.ttc', 'arial.ttf'):
    path = Path('C:/Windows/Fonts') / name
    if path.exists():
        QFontDatabase.addApplicationFont(str(path))
app.setFont(QFont('Microsoft YaHei', 10))
import videocaptioner.config as config
config.SETTINGS_PATH = Path(sys.argv[1]) / 'settings.json'
from videocaptioner.ui.common.config import cfg
from videocaptioner.core.entities import TranscribeModelEnum
from qfluentwidgets import setTheme, Theme
setTheme(Theme.LIGHT)
cfg.set(cfg.transcribe_model, TranscribeModelEnum.QWEN_FILETRANS)
from videocaptioner.ui.view.transcription_interface import TranscriptionInterface
from videocaptioner.ui.view.home_interface import HomeInterface
from videocaptioner.ui.view.setting_interface import SettingInterface

def settle():
    for _ in range(12):
        app.processEvents()

def visible_in_view(widget, scroll):
    top = widget.mapTo(scroll.viewport(), QPoint(0, 0)).y()
    assert top >= 0, (widget.objectName(), top)
    assert top + widget.height() <= scroll.viewport().height(), (top, widget.height(), scroll.viewport().height())

reports = []
for font_size, width, height in ((10, 1050, 650), (13, 900, 600), (12, 1200, 900)):
    app.setFont(QFont('Microsoft YaHei', font_size))
    host = HomeInterface()
    page = host.transcription_interface
    host.stackedWidget.setCurrentWidget(page)
    page.on_transcription_model_changed(TranscribeModelEnum.QWEN_FILETRANS.value)
    host.resize(width, height)
    host.show()
    settle()
    assert host.height() == height, (host.height(), height)
    assert page.video_info_card.transcription_interface is page
    scroll = page.content_scroll_area
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0
    qwen = page.transcription_setting_card.qwen_filetrans_widget
    qwen.hotwords_card.editor.setFocus()
    bar.setValue(0)
    settle()
    wheel = QWheelEvent(QPointF(5, 5), QPointF(scroll.viewport().mapToGlobal(QPoint(5, 5))),
                        QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    QApplication.sendEvent(scroll.viewport(), wheel)
    QTest.qWait(250)
    assert bar.value() > 0, 'wheel outside a focused editor must scroll the page'
    scroll.ensureWidgetVisible(qwen.context_card, 0, 0)
    settle()
    visible_in_view(qwen.context_card, scroll)
    bar.setValue(bar.maximum())
    settle()
    visible_in_view(qwen.check_card, scroll)
    assert page.command_bar.mapTo(page, QPoint(0, 0)).y() >= 0
    assert page.command_bar.isVisible()
    qwen.hotwords_card.editor.setPlainText('词条 | 6')
    settle()
    assert '权重' in qwen.hotwords_card.validationLabel.text()
    qwen.context_card.editor.setPlainText('甲' * 401)
    qwen.hotwords_card.editor.setPlainText('斯蒂格勒\n第三持存 | 5')
    settle()
    assert '400' in qwen.context_card.validationLabel.text()
    assert '权重' not in qwen.hotwords_card.validationLabel.text(), 'fixed hotwords must clear stale error even when context is invalid'
    qwen.context_card.editor.setPlainText('录音讨论斯蒂格勒的第三持存。')
    settle()
    qwen.validate_options()
    assert cfg.qwen_filetrans_hotwords.value == '斯蒂格勒\n第三持存 | 5'
    reports.append((width, height, font_size, bar.maximum()))
    page.on_transcription_model_changed(TranscribeModelEnum.MIMO_ASR.value)
    settle()
    assert host.height() == height
    mimo = page.transcription_setting_card.mimo_asr_widget
    assert mimo.scrollArea.viewport().height() >= 300, mimo.scrollArea.viewport().height()
    mimo.scrollArea.ensureWidgetVisible(mimo.check_connection_card, 0, 0)
    settle()
    visible_in_view(mimo.check_connection_card, mimo.scrollArea)
    page.on_transcription_model_changed(TranscribeModelEnum.QWEN_FILETRANS.value)
    settle()
    bar.setValue(bar.maximum())
    settle()
    visible_in_view(qwen.check_card, scroll)
    if width == 1050:
        assert host.grab().save(str(config.WORK_PATH / 'qwen-filetrans-home-bottom.png'))
        scroll.ensureWidgetVisible(qwen.hotwords_card, 0, 0)
        settle()
        assert host.grab().save(str(config.WORK_PATH / 'qwen-filetrans-home-hotwords.png'))
        bar.setValue(0)
        settle()
        assert host.grab().save(str(config.WORK_PATH / 'qwen-filetrans-home-top.png'))
    elif width == 900:
        assert host.grab().save(str(config.WORK_PATH / 'qwen-filetrans-home-narrow-bottom.png'))
    host.stackedWidget.setCurrentWidget(host.task_creation_interface)
    settle()
    assert host.stackedWidget.sizePolicy().verticalPolicy() == QSizePolicy.Preferred
    host.stackedWidget.setCurrentWidget(page)
    settle()
    host.resize(width, height)
    settle()
    assert host.height() == height
    host.close()
    settle()

standalone = TranscriptionInterface()
standalone.on_transcription_model_changed(TranscribeModelEnum.QWEN_FILETRANS.value)
standalone.resize(1050, 650)
standalone.show()
settle()
assert standalone.height() == 650
assert standalone.content_scroll_area.verticalScrollBar().maximum() > 0
standalone.content_scroll_area.verticalScrollBar().setValue(standalone.content_scroll_area.verticalScrollBar().maximum())
settle()
visible_in_view(standalone.transcription_setting_card.qwen_filetrans_widget.check_card, standalone.content_scroll_area)

settings = SettingInterface()
settings.resize(1050, 650)
palette = settings.scrollWidget.palette()
palette.setColor(QPalette.Window, QColor('#fafafa'))
settings.scrollWidget.setPalette(palette)
settings.scrollWidget.setAutoFillBackground(True)
settings.settingLabel.setStyleSheet('color: #202020; font: 33px "Microsoft YaHei";')
settings.show()
settle()
source = standalone.transcription_setting_card.qwen_filetrans_widget.context_card.editor
source.setPlainText('甲乙')
cursor = source.textCursor()
cursor.setPosition(1)
source.setTextCursor(cursor)
source.insertPlainText('中')
assert source.textCursor().position() == 2
source.insertPlainText('文')
assert source.toPlainText() == '甲中文乙' and source.textCursor().position() == 3
assert source.document().isUndoAvailable()
assert settings.qwenFileTransWidget.context_card.editor.toPlainText() == '甲中文乙'
source.undo()
assert source.toPlainText() != '甲中文乙'
source.setPlainText('录音讨论斯蒂格勒的第三持存。')
settle()
settings.ensureWidgetVisible(settings.qwenFileTransWidget.context_card, 0, 0)
settle()
visible_in_view(settings.qwenFileTransWidget.context_card, settings)
settings.ensureWidgetVisible(settings.qwenFileTransWidget.check_card, 0, 0)
settle()
visible_in_view(settings.qwenFileTransWidget.check_card, settings)
assert settings.grab().save(str(config.WORK_PATH / 'qwen-filetrans-global-settings.png'))
settings.close()
standalone.close()
settle()
saved = json.loads(config.SETTINGS_PATH.read_text(encoding='utf-8'))
assert saved['QwenFileTrans']['Hotwords'] == '斯蒂格勒\n第三持存 | 5'
assert saved['QwenFileTrans']['Context'] == '录音讨论斯蒂格勒的第三持存。'
print(json.dumps(reports))
"""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
