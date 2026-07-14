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

& $python -m PyInstaller --version *> $null
if ($LASTEXITCODE -ne 0) {
    uv pip install --python $python "pyinstaller>=6.0,<7.0"
}

& $python (Join-Path $projectRoot "scripts\create_app_icon.py")
if ($LASTEXITCODE -ne 0) {
    throw "Application icon generation failed."
}

Push-Location $projectRoot
try {
    & $python -m PyInstaller --noconfirm --clean --windowed --onedir `
        --name VideoCaptioner `
        --specpath build `
        --icon $icon `
        --add-data "$resourceDir;resource" `
        --add-data "$qwenRunner;videocaptioner\core\asr" `
        --collect-all qfluentwidgets `
        --collect-submodules modelscope `
        videocaptioner\ui\main.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
    if (-not (Test-Path $qwenRunnerBundlePath)) {
        throw "PyInstaller did not bundle the Qwen3-ASR runner: $qwenRunnerBundlePath"
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
