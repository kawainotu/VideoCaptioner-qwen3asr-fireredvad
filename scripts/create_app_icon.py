"""Create a multi-resolution Windows icon from the canonical application logo."""

from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "resource" / "assets" / "logo.png"
OUTPUT = SOURCE.with_suffix(".ico")
SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> None:
    with Image.open(SOURCE) as image:
        image.convert("RGBA").save(OUTPUT, format="ICO", sizes=[(size, size) for size in SIZES])


if __name__ == "__main__":
    main()
