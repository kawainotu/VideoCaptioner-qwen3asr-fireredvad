"""Download the inference-only FireRedVAD files used by packaged builds."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    snapshot_download(
        repo_id="FireRedTeam/FireRedVAD",
        local_dir=args.destination,
        allow_patterns=["README.md", "VAD/*"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
