[CmdletBinding()]
param(
    [string]$Version = "1.4.2"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$resourceDir = Join-Path $projectRoot "resource"
$icon = Join-Path $resourceDir "assets\logo.ico"
$qwenRunner = Join-Path $projectRoot "videocaptioner\core\asr\qwen3_asr_runner.py"
$qwenRunnerBundlePath = Join-Path $projectRoot "dist\VideoCaptioner\_internal\videocaptioner\core\asr\qwen3_asr_runner.py"
$mimoVadRunner = Join-Path $projectRoot "videocaptioner\core\asr\mimo_vad_runner.py"
$mimoAlignmentRunner = Join-Path $projectRoot "videocaptioner\core\asr\mimo_alignment_runner.py"
$mimoVadDefaults = Join-Path $projectRoot "videocaptioner\core\mimo_vad_defaults.py"
$mimoVadRunnerBundlePath = Join-Path $projectRoot "dist\VideoCaptioner\_internal\videocaptioner\core\asr\mimo_vad_runner.py"
$mimoAlignmentRunnerBundlePath = Join-Path $projectRoot "dist\VideoCaptioner\_internal\videocaptioner\core\asr\mimo_alignment_runner.py"
$mimoVadDefaultsBundlePath = Join-Path $projectRoot "dist\VideoCaptioner\_internal\videocaptioner\core\mimo_vad_defaults.py"
$fireredVadModelDir = Join-Path $projectRoot "build\fireredvad-model"
$fireredVadWeights = Join-Path $fireredVadModelDir "VAD\model.pth.tar"
$fireredVadBundlePath = Join-Path $projectRoot "dist\VideoCaptioner\_internal\models\FireRedVAD\VAD\model.pth.tar"
$bundledQwenPythonDir = Join-Path $projectRoot "dist\VideoCaptioner\_internal\qwen-python"
$bundledQwenPython = Join-Path $bundledQwenPythonDir "python.exe"
$iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $innoCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
    )
    $iscc = $innoCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found: $python"
}
if (-not (Test-Path $qwenRunner)) {
    throw "Qwen3-ASR runner not found in source tree: $qwenRunner"
}
foreach ($mimoSource in @($mimoVadRunner, $mimoAlignmentRunner, $mimoVadDefaults)) {
    if (-not (Test-Path $mimoSource)) {
        throw "MiMo isolated worker source not found: $mimoSource"
    }
}

$pythonMetadata = @(& $python -c "import sys; print(sys.base_prefix); print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ($LASTEXITCODE -ne 0 -or $pythonMetadata.Count -ne 2) {
    throw "Could not determine the base Python distribution used by $python"
}
$basePythonDir = $pythonMetadata[0].Trim()
$basePythonVersion = $pythonMetadata[1].Trim()
$basePythonExe = Join-Path $basePythonDir "python.exe"
if ($basePythonVersion -notin @("3.10", "3.11", "3.12")) {
    throw "The installer must be built with Python 3.10, 3.11, or 3.12; found $basePythonVersion"
}
if (-not (Test-Path $basePythonExe)) {
    throw "Base Python executable not found: $basePythonExe"
}

if (-not (Test-Path $fireredVadWeights)) {
    & $python (Join-Path $projectRoot "scripts\download_firered_vad.py") $fireredVadModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "FireRedVAD model download failed."
    }
}
if (-not (Test-Path $fireredVadWeights)) {
    throw "FireRedVAD weights not found after download: $fireredVadWeights"
}

$hasPyInstaller = $false
try {
    & $python -m PyInstaller --version *> $null
    $hasPyInstaller = $LASTEXITCODE -eq 0
}
catch {
    # PowerShell treats a missing module as a terminating native-command error with ErrorActionPreference=Stop.
    $hasPyInstaller = $false
}
if (-not $hasPyInstaller) {
    uv pip install --python $python "pyinstaller>=6.0,<7.0"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller installation failed."
    }
}

& $python (Join-Path $projectRoot "scripts\create_app_icon.py")
if ($LASTEXITCODE -ne 0) {
    throw "Application icon generation failed."
}

Push-Location $projectRoot
try {
    # Do not inherit stale editable-install / VCS metadata in the frozen app.
    $versionSource = Join-Path $projectRoot "videocaptioner\_version.py"
    $writeBuildVersion = @'
import re
import sys
from pathlib import Path

version = sys.argv[1]
if not re.fullmatch(r"\d+\.\d+\.\d+", version):
    raise SystemExit("Installer version must have three numeric components")
version_tuple = tuple(int(part) for part in version.split("."))
source = (
    "# Generated for the installer build; do not track.\n"
    "__all__ = ['__version__', '__version_tuple__', 'version', 'version_tuple', '__commit_id__', 'commit_id']\n"
    f"__version__ = version = {version!r}\n"
    f"__version_tuple__ = version_tuple = {version_tuple!r}\n"
    "__commit_id__ = commit_id = None\n"
)
Path(sys.argv[2]).write_text(source, encoding="utf-8")
'@
    & $python -c $writeBuildVersion $Version $versionSource
    if ($LASTEXITCODE -ne 0) {
        throw "Could not generate application version metadata for $Version"
    }
    & $python -m PyInstaller --noconfirm --clean --windowed --onedir `
        --name VideoCaptioner `
        --specpath build `
        --icon $icon `
        --add-data "$resourceDir;resource" `
        --add-data "$qwenRunner;videocaptioner\core\asr" `
        --add-data "$mimoVadRunner;videocaptioner\core\asr" `
        --add-data "$mimoAlignmentRunner;videocaptioner\core\asr" `
        --add-data "$mimoVadDefaults;videocaptioner\core" `
        --add-data "$fireredVadModelDir;models\FireRedVAD" `
        --collect-all qfluentwidgets `
        --collect-submodules modelscope `
        videocaptioner\ui\main.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
    if (-not (Test-Path $qwenRunnerBundlePath)) {
        throw "PyInstaller did not bundle the Qwen3-ASR runner: $qwenRunnerBundlePath"
    }
    if (-not (Test-Path $fireredVadBundlePath)) {
        throw "PyInstaller did not bundle FireRedVAD: $fireredVadBundlePath"
    }
    foreach ($mimoBundledSource in @($mimoVadRunnerBundlePath, $mimoAlignmentRunnerBundlePath, $mimoVadDefaultsBundlePath)) {
        if (-not (Test-Path $mimoBundledSource)) {
            throw "PyInstaller did not bundle MiMo isolated worker source: $mimoBundledSource"
        }
    }
    if (Test-Path $bundledQwenPythonDir) {
        throw "PyInstaller output unexpectedly contains the bundled Qwen Python directory: $bundledQwenPythonDir"
    }

    New-Item -ItemType Directory -Path $bundledQwenPythonDir | Out-Null
    Copy-Item -Path (Join-Path $basePythonDir "*") -Destination $bundledQwenPythonDir -Recurse -Force
    if (-not (Test-Path $bundledQwenPython)) {
        throw "Bundled Qwen Python executable was not copied: $bundledQwenPython"
    }
    & $bundledQwenPython -c "import ensurepip, sys, venv; assert sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)); print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw "Bundled Qwen Python is not a usable compatible CPython runtime."
    }

    if (-not $iscc) {
        throw "Inno Setup 6 is required. Install it with: winget install --id JRSoftware.InnoSetup --exact"
    }

    & $iscc "/DMyAppVersion=$Version" "installer\VideoCaptioner.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup build failed."
    }
}
finally {
    Pop-Location
}
