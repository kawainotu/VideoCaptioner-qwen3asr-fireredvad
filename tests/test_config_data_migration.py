from videocaptioner.config import _migrate_legacy_appdata


def test_migrate_legacy_appdata_moves_models_runtimes_and_settings(tmp_path):
    appdata = tmp_path / "VideoCaptioner"
    legacy = appdata / "VideoCaptioner"
    model = legacy / "models" / "Qwen3-ASR-1.7B"
    runtime = legacy / "runtimes" / "qwen3-asr"
    model.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (runtime / "marker").write_text("ready", encoding="utf-8")
    (legacy / "settings.json").write_text("{}", encoding="utf-8")

    _migrate_legacy_appdata(legacy, appdata)

    assert (appdata / "models" / "Qwen3-ASR-1.7B" / "config.json").is_file()
    assert (appdata / "runtimes" / "qwen3-asr" / "marker").is_file()
    assert (appdata / "settings.json").is_file()
    assert not (legacy / "models" / "Qwen3-ASR-1.7B").exists()
    assert not (legacy / "runtimes" / "qwen3-asr").exists()


def test_migrate_legacy_appdata_merges_missing_entries_without_overwriting(tmp_path):
    appdata = tmp_path / "VideoCaptioner"
    legacy = appdata / "VideoCaptioner"
    source_model = legacy / "models"
    destination_model = appdata / "models"
    source_model.mkdir(parents=True)
    destination_model.mkdir(parents=True)
    (source_model / "from-legacy").write_text("legacy", encoding="utf-8")
    (destination_model / "existing").write_text("keep", encoding="utf-8")

    _migrate_legacy_appdata(legacy, appdata)

    assert (destination_model / "existing").read_text(encoding="utf-8") == "keep"
    assert (destination_model / "from-legacy").read_text(encoding="utf-8") == "legacy"
    assert not (source_model / "from-legacy").exists()
