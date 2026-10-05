import datetime
import tempfile
import threading
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal

from videocaptioner.core.asr import transcribe
from videocaptioner.core.entities import (
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
    TranscribeTask,
)
from videocaptioner.core.utils.logger import setup_logger
from videocaptioner.core.utils.video_utils import video2audio

logger = setup_logger("transcript_thread")
_ACTIVE_TRANSCRIPT_THREADS = set()
_LAST_CONNECTED_APP = None


def cancel_active_mimo_transcriptions():
    """Stop MiMo runners even when the app quits without closing child pages."""
    for thread in tuple(_ACTIVE_TRANSCRIPT_THREADS):
        config = thread.task.transcribe_config
        if config and config.transcribe_model == TranscribeModelEnum.MIMO_ASR:
            thread.cancel()


def _connect_app_shutdown():
    global _LAST_CONNECTED_APP
    app = QCoreApplication.instance()
    if app is not None and app is not _LAST_CONNECTED_APP:
        app.aboutToQuit.connect(cancel_active_mimo_transcriptions)
        _LAST_CONNECTED_APP = app


class TranscriptThread(QThread):
    finished = pyqtSignal(TranscribeTask)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, task: TranscribeTask):
        super().__init__()
        self.task = task
        self._cancelled = threading.Event()
        self._asr = None
        _connect_app_shutdown()
        QThread.finished.__get__(self, type(self)).connect(self._release_worker)

    def start(self, *args, **kwargs):
        _ACTIVE_TRANSCRIPT_THREADS.add(self)
        super().start(*args, **kwargs)

    def _release_worker(self):
        _ACTIVE_TRANSCRIPT_THREADS.discard(self)

    def cancel(self):
        """Request cancellation without blocking the UI or bypassing cleanup."""
        self._cancelled.set()
        asr = self._asr
        if asr is not None and hasattr(asr, "cancel"):
            asr.cancel()

    def _on_asr_created(self, asr):
        self._asr = asr
        if self._cancelled.is_set():
            if hasattr(asr, "cancel"):
                asr.cancel()
            self._raise_if_cancelled()

    def _raise_if_cancelled(self):
        if self._cancelled.is_set():
            raise RuntimeError("转录任务已取消")

    def run(self):
        try:
            self._raise_if_cancelled()
            self.task.started_at = datetime.datetime.now()
            logger.info(f"\n{self.task.transcribe_config.print_config()}")

            self._validate_task()

            # 检查是否已下载字幕文件
            if self._check_downloaded_subtitle():
                return

            self._perform_transcription()

        except Exception as e:
            if self._cancelled.is_set():
                return
            logger.exception("转录过程中发生错误: %s", str(e))
            self.error.emit(str(e))
            self.progress.emit(100, self.tr("转录失败"))
        finally:
            self._asr = None

    def _validate_task(self):
        """验证任务配置"""
        if not self.task.file_path:
            raise ValueError(self.tr("文件路径为空"))

        video_path = Path(self.task.file_path)
        if not video_path.exists():
            logger.error(f"视频文件不存在：{video_path}")
            raise ValueError(self.tr("视频文件不存在"))

        if not self.task.transcribe_config:
            raise ValueError(self.tr("转录配置为空"))

        if not self.task.output_path:
            raise ValueError(self.tr("输出路径为空"))

    def _check_downloaded_subtitle(self) -> bool:
        """检查是否存在下载的字幕文件"""
        if not (self.task.need_next_task and self.task.file_path):
            return False

        subtitle_dir = Path(self.task.file_path).parent / "subtitle"
        if not subtitle_dir.exists():
            return False

        downloaded_subtitles = list(subtitle_dir.glob("【下载字幕】*"))
        if not downloaded_subtitles:
            return False

        subtitle_file = downloaded_subtitles[0]
        self._raise_if_cancelled()
        self.task.output_path = str(subtitle_file)
        logger.info(f"字幕文件已下载，跳过转录。找到下载的字幕文件：{subtitle_file}")
        self.progress.emit(100, self.tr("字幕已下载"))
        self.finished.emit(self.task)
        return True

    def _perform_transcription(self):
        """执行转录流程"""
        assert self.task.file_path is not None
        assert self.task.transcribe_config is not None
        assert self.task.output_path is not None

        video_path = Path(self.task.file_path)

        self.progress.emit(5, self.tr("转换音频中"))
        logger.info("开始转换音频")

        # 创建临时音频文件（delete=False 避免 Windows 权限问题）
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()  # 立即关闭文件句柄，让 ffmpeg 可以写入

        try:
            # 转换音频文件
            # 获取选中的音轨索引（如果有）
            audio_track_index = self.task.selected_audio_track_index
            is_success = video2audio(
                str(video_path),
                output=temp_audio_path,
                audio_track_index=audio_track_index,
            )
            self._raise_if_cancelled()
            if not is_success:
                logger.error("音频转换失败")
                raise RuntimeError(self.tr("音频转换失败"))

            self.progress.emit(20, self.tr("语音转录中"))
            logger.info("开始语音转录")

            # 进行转录
            asr_data = transcribe(
                temp_audio_path,
                self.task.transcribe_config,
                callback=self.progress_callback,
                on_asr_created=self._on_asr_created,
            )
            self._raise_if_cancelled()

            # 保存字幕文件（根据配置的输出格式）
            output_path = Path(self.task.output_path)
            output_format_enum = self.task.transcribe_config.output_format
            base_path = output_path.with_suffix("")

            # 根据选择的格式导出
            if output_format_enum == TranscribeOutputFormatEnum.ALL:
                formats_to_export = [
                    fmt.value.lower()
                    for fmt in TranscribeOutputFormatEnum
                    if fmt != TranscribeOutputFormatEnum.ALL
                ]
            else:
                formats_to_export = [output_format_enum.value.lower()]

            if self.task.need_next_task:
                formats_to_export.append(TranscribeOutputFormatEnum.SRT.value.lower())
            formats_to_export = list(set(formats_to_export))

            # 保存字幕文件
            for fmt in formats_to_export:
                self._raise_if_cancelled()
                save_path = f"{base_path}.{fmt}"
                asr_data.save(save_path)
                logger.info("%s 字幕文件已保存到: %s", fmt.upper(), save_path)

            self._raise_if_cancelled()
            self.progress.emit(100, self.tr("转录完成"))
            self.finished.emit(self.task)
        finally:
            Path(temp_audio_path).unlink(missing_ok=True)

    def progress_callback(self, value, message):
        self._raise_if_cancelled()
        progress = min(20 + (value * 0.8), 100)
        self.progress.emit(int(progress), message)
