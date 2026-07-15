import videocaptioner.ui.thread.huggingface_download_thread as download_module
from videocaptioner.ui.thread.huggingface_download_thread import HuggingFaceDownloadThread


def test_huggingface_download_uses_mirror_then_official_fallback(monkeypatch, tmp_path):
    endpoints = []

    def snapshot_download(*, endpoint, **_kwargs):
        endpoints.append(endpoint)
        if endpoint == "https://hf-mirror.com":
            raise RuntimeError("mirror unavailable")

    monkeypatch.delenv("VIDEOCAPTIONER_HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(download_module, "snapshot_download", snapshot_download)
    thread = HuggingFaceDownloadThread("owner/model", str(tmp_path / "model"))
    errors = []
    thread.error.connect(errors.append)

    thread.run()

    assert endpoints == ["https://hf-mirror.com", "https://huggingface.co"]
    assert errors == []
