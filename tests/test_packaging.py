from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_installer_build_bundles_qwen3_asr_runner():
    script = (PROJECT_ROOT / "scripts" / "build_installer.ps1").read_text(encoding="utf-8")

    assert '$qwenRunner = Join-Path $projectRoot "videocaptioner\\core\\asr\\qwen3_asr_runner.py"' in script
    assert '--add-data "$qwenRunner;videocaptioner\\core\\asr"' in script
    assert "$qwenRunnerBundlePath" in script
    assert "scripts\\download_firered_vad.py" in script
    assert '--add-data "$fireredVadModelDir;models\\FireRedVAD"' in script
    assert "$fireredVadBundlePath" in script
    assert '$bundledQwenPythonDir = Join-Path $projectRoot "dist\\VideoCaptioner\\_internal\\qwen-python"' in script
    assert "sys.base_prefix" in script
    assert "Copy-Item -Path (Join-Path $basePythonDir \"*\")" in script
    assert "$bundledQwenPython" in script


def test_installer_build_bootstraps_pyinstaller_when_missing():
    script = (PROJECT_ROOT / "scripts" / "build_installer.ps1").read_text(encoding="utf-8")

    assert "$hasPyInstaller = $false" in script
    assert 'uv pip install --python $python "pyinstaller>=6.0,<7.0"' in script


def test_installer_filename_includes_platform_version_and_bundled_features():
    installer_script = (PROJECT_ROOT / "installer" / "VideoCaptioner.iss").read_text(encoding="utf-8")

    assert '#define MyAppArchitecture "win64"' in installer_script
    assert '#define MyAppFeatures "qwen3-asr-fireredvad-1.2"' in installer_script
    assert (
        "OutputBaseFilename={#MyAppName}-Setup-{#MyAppArchitecture}-v{#MyAppVersion}-{#MyAppFeatures}"
        in installer_script
    )
