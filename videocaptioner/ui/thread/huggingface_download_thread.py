from collections.abc import Callable

from huggingface_hub import snapshot_download
from PyQt5.QtCore import QThread, pyqtSignal
from tqdm.auto import tqdm


def create_progress_bar_class(
    progress_callback: Callable[[int, str], None],
) -> type[tqdm]:
    """Create a quiet tqdm implementation that forwards progress to the UI."""

    class ProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            self._progress_callback = progress_callback
            self._last_percentage = -1
            self._description = str(kwargs.get("desc", ""))
            super().__init__(*args, disable=True, **kwargs)
            self._emit_progress()

        def update(self, n=1):
            self.n += n or 0
            self._emit_progress()

        def refresh(self, *args, **kwargs):
            self._emit_progress()

        def set_description(self, desc=None, refresh=True):
            self._description = str(desc or "")
            if refresh:
                self._emit_progress()

        def _emit_progress(self) -> None:
            if not self.total:
                return
            percentage = min(int(self.n * 100 / self.total), 99)
            if percentage == self._last_percentage:
                return
            self._last_percentage = percentage
            description = self._description or "Downloading model"
            self._progress_callback(percentage, f"{description}: {percentage}%")

    return ProgressBar


class HuggingFaceDownloadThread(QThread):
    """Download a model snapshot from Hugging Face into a selected folder."""

    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        model_id: str,
        save_path: str,
        ignore_patterns: tuple[str, ...] = (),
    ):
        super().__init__()
        self.model_id = model_id
        self.save_path = save_path
        self.ignore_patterns = ignore_patterns

    def run(self) -> None:
        try:
            self.progress.emit(0, self.tr("正在准备下载"))
            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.save_path,
                ignore_patterns=list(self.ignore_patterns) or None,
                tqdm_class=create_progress_bar_class(self.progress.emit),
            )
            self.progress.emit(100, self.tr("下载完成"))
        except Exception as exc:
            self.error.emit(str(exc))
