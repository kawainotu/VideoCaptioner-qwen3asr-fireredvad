import logging
import os
import shutil
import sys
from pathlib import Path

try:
    from videocaptioner._version import __version__ as _raw_version
    # Strip dev suffix (e.g. "1.5.0.dev103+g38544177c" → "1.5.0")
    VERSION = _raw_version.split(".dev")[0]
except Exception:
    VERSION = "0.0.0-dev"
YEAR = 2026
APP_NAME = "VideoCaptioner"
AUTHOR = "Weifeng"

HELP_URL = "https://github.com/WEIFENG2333/VideoCaptioner"
GITHUB_REPO_URL = "https://github.com/WEIFENG2333/VideoCaptioner"
RELEASE_URL = "https://github.com/WEIFENG2333/VideoCaptioner/releases/latest"
FEEDBACK_URL = "https://github.com/WEIFENG2333/VideoCaptioner/issues"

# Detect whether running from source tree or pip-installed
_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

# Development mode: resource/ exists next to the package
_IS_DEV = (_PROJECT_ROOT / "resource").is_dir() and not getattr(sys, "frozen", False)


def _migrate_legacy_appdata(legacy_path: Path, appdata_path: Path) -> None:
    """Move data out of the former duplicated Windows app-data directory.

    ``platformdirs.user_data_dir(APP_NAME)`` uses the application name as both
    the author and application directory on Windows. Earlier packaged builds
    therefore stored data in ``.../VideoCaptioner/VideoCaptioner`` while older
    releases used ``.../VideoCaptioner``. Only move entries that do not yet
    exist at the canonical destination so an upgrade never overwrites user data.
    """
    if legacy_path == appdata_path or not legacy_path.is_dir():
        return

    appdata_path.mkdir(parents=True, exist_ok=True)
    for name in ("models", "runtimes", "cache", "logs"):
        source_directory = legacy_path / name
        destination_directory = appdata_path / name
        if not source_directory.is_dir():
            continue
        destination_directory.mkdir(parents=True, exist_ok=True)
        for source in source_directory.iterdir():
            destination = destination_directory / source.name
            if destination.exists():
                continue
            try:
                shutil.move(str(source), str(destination))
            except (OSError, shutil.Error) as exc:
                logging.getLogger(__name__).warning(
                    "Could not migrate VideoCaptioner data from %s to %s: %s",
                    source,
                    destination,
                    exc,
                )

    source_settings = legacy_path / "settings.json"
    destination_settings = appdata_path / "settings.json"
    if source_settings.is_file() and not destination_settings.exists():
        try:
            shutil.move(str(source_settings), str(destination_settings))
        except (OSError, shutil.Error) as exc:
            logging.getLogger(__name__).warning(
                "Could not migrate VideoCaptioner settings from %s to %s: %s",
                source_settings,
                destination_settings,
                exc,
            )

if _IS_DEV:
    ROOT_PATH = _PROJECT_ROOT
    RESOURCE_PATH = ROOT_PATH / "resource"
    APPDATA_PATH = ROOT_PATH / "AppData"
    WORK_PATH = ROOT_PATH / "work-dir"
else:
    # Installed via pip — use platform-appropriate directories
    from platformdirs import user_data_dir

    # appauthor=False keeps the established Windows location as
    # %LOCALAPPDATA%/VideoCaptioner instead of nesting the app name twice.
    APPDATA_PATH = Path(user_data_dir(APP_NAME, appauthor=False))
    _migrate_legacy_appdata(Path(user_data_dir(APP_NAME)), APPDATA_PATH)
    if getattr(sys, "frozen", False):
        ROOT_PATH = Path(getattr(sys, "_MEIPASS", _PROJECT_ROOT))
        RESOURCE_PATH = ROOT_PATH / "resource"
    else:
        ROOT_PATH = APPDATA_PATH
        RESOURCE_PATH = ROOT_PATH / "resource"
    WORK_PATH = Path.home() / "VideoCaptioner"

BIN_PATH = RESOURCE_PATH / "bin"
ASSETS_PATH = RESOURCE_PATH / "assets"
SUBTITLE_STYLE_PATH = RESOURCE_PATH / "subtitle_style"
TRANSLATIONS_PATH = RESOURCE_PATH / "translations"
FONTS_PATH = RESOURCE_PATH / "fonts"

# Fallback: bundled fonts inside the package (for pip install)
_BUNDLED_FONTS = _PACKAGE_DIR / "resources" / "fonts"
if not FONTS_PATH.exists() and _BUNDLED_FONTS.exists():
    FONTS_PATH = _BUNDLED_FONTS

LOG_PATH = APPDATA_PATH / "logs"
LLM_LOG_FILE = LOG_PATH / "llm_requests.jsonl"
SETTINGS_PATH = APPDATA_PATH / "settings.json"
CACHE_PATH = APPDATA_PATH / "cache"
MODEL_PATH = APPDATA_PATH / "models"
RUNTIME_PATH = APPDATA_PATH / "runtimes"

FASTER_WHISPER_PATH = BIN_PATH / "Faster-Whisper-XXL"
QWEN3_ASR_RUNTIME_PATH = RUNTIME_PATH / "qwen3-asr"
QWEN3_ASR_MODEL_PATH = MODEL_PATH / "Qwen3-ASR-1.7B"
QWEN3_ALIGNER_MODEL_PATH = MODEL_PATH / "Qwen3-ForcedAligner-0.6B"
FIRERED_VAD_MODEL_PATH = MODEL_PATH / "FireRedVAD"
BUNDLED_FIRERED_VAD_MODEL_PATH = ROOT_PATH / "models" / "FireRedVAD"

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Add bin paths to PATH (only if they exist)
if BIN_PATH.exists():
    os.environ["PATH"] = str(FASTER_WHISPER_PATH) + os.pathsep + os.environ["PATH"]
    os.environ["PATH"] = str(BIN_PATH) + os.pathsep + os.environ["PATH"]

if (BIN_PATH / "vlc").exists():
    os.environ["PYTHON_VLC_MODULE_PATH"] = str(BIN_PATH / "vlc")

# Create data directories
for p in [CACHE_PATH, LOG_PATH, WORK_PATH, MODEL_PATH, RUNTIME_PATH]:
    p.mkdir(parents=True, exist_ok=True)
