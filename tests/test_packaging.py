import shutil
import subprocess
import sys
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


def test_installer_bundles_all_mimo_direct_execution_sources(tmp_path):
    script = (PROJECT_ROOT / "scripts/build_installer.ps1").read_text(encoding="utf-8")
    for variable, source, destination in (
        ("mimoVadRunner", "asr/mimo_vad_runner.py", "videocaptioner\\core\\asr"),
        ("mimoAlignmentRunner", "asr/mimo_alignment_runner.py", "videocaptioner\\core\\asr"),
        ("mimoVadDefaults", "mimo_vad_defaults.py", "videocaptioner\\core"),
    ):
        assert f'--add-data "${variable};{destination}"' in script
        assert f"${variable}BundlePath" in script
        target = tmp_path / "videocaptioner/core" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / "videocaptioner/core" / source, target)
    for runner in ("mimo_vad_runner.py", "mimo_alignment_runner.py"):
        result = subprocess.run(
            [sys.executable, str(tmp_path / "videocaptioner/core/asr" / runner), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr


def test_installer_generates_release_version_before_pyinstaller(tmp_path):
    script = (PROJECT_ROOT / "scripts/build_installer.ps1").read_text(encoding="utf-8")
    code = script.split("$writeBuildVersion = @'\n", 1)[1].split("\n'@", 1)[0]
    output = tmp_path / "_version.py"
    result = subprocess.run([sys.executable, "-c", code, "1.4.2", str(output)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    namespace = {}
    exec(output.read_text(encoding="utf-8"), namespace)
    assert namespace["__version__"] == namespace["version"] == "1.4.2"
    assert namespace["__version_tuple__"] == namespace["version_tuple"] == (1, 4, 2)
    assert namespace["__commit_id__"] == namespace["commit_id"] is None
    assert script.index("& $python -c $writeBuildVersion") < script.index("& $python -m PyInstaller --noconfirm")


def test_installer_filename_includes_platform_version_and_bundled_features():
    installer_script = (PROJECT_ROOT / "installer" / "VideoCaptioner.iss").read_text(encoding="utf-8")

    assert '#define MyAppArchitecture "win64"' in installer_script
    assert '#define MyAppFeatures "qwen3-asr-fireredvad-1.3"' in installer_script
    assert (
        "OutputBaseFilename={#MyAppName}-Setup-{#MyAppArchitecture}-v{#MyAppVersion}-{#MyAppFeatures}"
        in installer_script
    )
