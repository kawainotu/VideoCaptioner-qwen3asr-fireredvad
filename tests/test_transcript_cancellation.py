"""Cancellation tests run methods directly without any Qt application or GUI."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from videocaptioner.core.entities import TranscribeConfig, TranscribeModelEnum, TranscribeTask
from videocaptioner.ui.thread import transcript_thread as module
from videocaptioner.ui.thread.transcript_thread import TranscriptThread


class TranscriptCancellationTests(unittest.TestCase):
    def make_thread(self, file_path="audio.wav", output_path="output.srt"):
        return TranscriptThread(TranscribeTask(
            file_path=file_path, output_path=output_path,
            transcribe_config=TranscribeConfig(transcribe_model=TranscribeModelEnum.MIMO_ASR),
        ))

    def test_cancel_reaches_current_asr(self):
        thread = self.make_thread()
        asr = Mock()
        thread._on_asr_created(asr)
        thread.cancel()
        asr.cancel.assert_called_once_with()
        self.assertTrue(thread._cancelled.is_set())

    def test_cancel_before_asr_creation_prevents_run(self):
        thread = self.make_thread()
        thread.cancel()
        asr = Mock()
        with self.assertRaisesRegex(RuntimeError, "取消"):
            thread._on_asr_created(asr)
        asr.cancel.assert_called_once_with()

    def test_app_shutdown_only_cancels_active_mimo(self):
        mimo = self.make_thread()
        other = self.make_thread()
        other.task.transcribe_config.transcribe_model = TranscribeModelEnum.QWEN3_ASR
        mimo.cancel = Mock()
        other.cancel = Mock()
        with patch.object(module, "_ACTIVE_TRANSCRIPT_THREADS", {mimo, other}):
            module.cancel_active_mimo_transcriptions()
        mimo.cancel.assert_called_once_with()
        other.cancel.assert_not_called()

    def test_shutdown_hook_connects_once_per_application(self):
        app = Mock()
        with patch.object(module, "_LAST_CONNECTED_APP", None), patch.object(module.QCoreApplication, "instance", return_value=app):
            module._connect_app_shutdown()
            module._connect_app_shutdown()
        app.aboutToQuit.connect.assert_called_once_with(module.cancel_active_mimo_transcriptions)

    def test_cancelled_result_is_not_saved_or_reported_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"test")
            output = Path(directory) / "out.srt"
            thread = self.make_thread(str(source), str(output))
            result = Mock()
            finished = Mock()
            error = Mock()
            thread.finished.connect(finished)
            thread.error.connect(error)

            def cancelled_transcription(*args, **kwargs):
                kwargs["on_asr_created"](Mock())
                thread.cancel()
                return result

            with patch("videocaptioner.ui.thread.transcript_thread.video2audio", return_value=True), patch(
                "videocaptioner.ui.thread.transcript_thread.transcribe", side_effect=cancelled_transcription
            ):
                thread.run()
            result.save.assert_not_called()
            finished.assert_not_called()
            error.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIsNone(thread._asr)

    def test_failure_error_is_last_so_ui_keeps_retry_state(self):
        thread = self.make_thread()
        events = []
        state = {"button": "正在转录", "enabled": False}

        def on_progress(value, message):
            events.append(("progress", value))
            state["button"] = "正在转录"

        def on_error(message):
            events.append(("error", message))
            state.update(button="重新转录", enabled=True)

        thread.progress.connect(on_progress)
        thread.error.connect(on_error)
        finished = Mock()
        thread.finished.connect(finished)
        with patch.object(thread, "_validate_task", side_effect=ValueError("alignment failed")):
            thread.run()
        self.assertEqual(events, [("progress", 100), ("error", "alignment failed")])
        self.assertEqual(state, {"button": "重新转录", "enabled": True})
        finished.assert_not_called()


if __name__ == "__main__":
    unittest.main()
