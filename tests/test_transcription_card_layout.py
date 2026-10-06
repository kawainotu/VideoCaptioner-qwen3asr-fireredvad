"""Exercise status updates without importing Qt or creating an application."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).parents[1] / "videocaptioner/ui/view/transcription_interface.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
CARD = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "VideoInfoCard")


def method(name):
    node = next(node for node in CARD.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Optional": __import__("typing").Optional, "PillPushButton": object,
                 "InfoBar": Mock(), "INFOBAR_DURATION_ERROR": 1}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class TranscriptionCardStatusTests(unittest.TestCase):
    def setUp(self):
        self.card = SimpleNamespace(
            start_button=Mock(), status_label=Mock(), button_widget=Mock(),
            progress_ring=Mock(), finished=Mock(), tr=lambda text: text,
            transcription_interface=SimpleNamespace(is_processing=True),
            parent=Mock(), window=Mock(),
        )
        self.card.set_status_text = lambda *args: method("set_status_text")(self.card, *args)

    def test_full_long_progress_is_preserved_outside_action(self):
        message = "MiMo 正在处理第 99/120 段：" + "很长的处理信息" * 80
        method("on_transcript_progress")(self.card, 82, message)
        self.card.start_button.setText.assert_called_once_with("正在转录")
        self.card.status_label.setText.assert_called_once_with(message)
        self.card.start_button.setToolTip.assert_called_once_with(message)
        self.card.progress_ring.setValue.assert_called_once_with(82)
        self.assertEqual(self.card.button_widget.mock_calls, [])

    def test_reset_clears_progress_and_tooltips(self):
        self.card.set_status_text("old progress")
        method("reset_ui")(self.card)
        self.card.start_button.setText.assert_called_with("开始转录")
        self.card.status_label.setText.assert_called_with("")
        self.card.status_label.setVisible.assert_called_with(False)
        self.card.start_button.setToolTip.assert_called_with("")
        self.card.progress_ring.hide.assert_called_once()

    def test_completion_replaces_progress(self):
        task = object()
        method("on_transcript_finished")(self.card, task)
        self.card.start_button.setText.assert_called_with("转录完成")
        self.card.status_label.setText.assert_called_with("转录完成")
        self.card.finished.emit.assert_called_once_with(task)

    def test_failure_preserves_error_and_restores_retry_action(self):
        method("on_transcript_error")(self.card, "MiMo request failed")
        self.card.start_button.setText.assert_called_with("重新转录")
        self.card.status_label.setText.assert_called_with("转录失败: MiMo request failed")
        self.assertFalse(self.card.transcription_interface.is_processing)
        self.card.start_button.setEnabled.assert_called_once_with(True)

    def test_metadata_width_follows_complete_new_content(self):
        for text in ("画质: 3840x2160", "大小: 5105.9 MB", "时长: 1:12:59"):
            button = Mock()
            button.sizeHint.return_value.width.return_value = 140
            method("set_metadata_text")(self.card, button, text)
            button.setText.assert_called_once_with(text)
            button.setMinimumWidth.assert_called_once_with(140)
            button.setToolTip.assert_called_once_with(text)


if __name__ == "__main__":
    unittest.main()
