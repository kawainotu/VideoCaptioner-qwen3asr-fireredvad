from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_installer_build_bundles_qwen3_asr_runner():
    script = (PROJECT_ROOT / "scripts" / "build_installer.ps1").read_text(encoding="utf-8")

    assert '$qwenRunner = Join-Path $projectRoot "videocaptioner\\core\\asr\\qwen3_asr_runner.py"' in script
    assert '--add-data "$qwenRunner;videocaptioner\\core\\asr"' in script
    assert "$qwenRunnerBundlePath" in script
