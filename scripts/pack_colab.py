"""Zip src/ and colab/ into dist/colab_bundle.zip. Allow-listed file types only, so no data can ride along."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "dist" / "colab_bundle.zip"
ALLOWED = {".py", ".ipynb", ".md"}


def bundle_files() -> list:
    files = [p for d in ("src", "colab") for p in sorted((ROOT / d).rglob("*"))
             if p.is_file() and p.suffix in ALLOWED and "__pycache__" not in p.parts
             and ".ipynb_checkpoints" not in p.parts]
    for p in files:
        if p.suffix == ".ipynb":
            cells = json.loads(p.read_text(encoding="utf-8"))["cells"]
            assert not any(c.get("outputs") for c in cells), f"clear outputs first: {p.name}"
    return files


def main():
    files = bundle_files()
    BUNDLE.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
    print(f"{BUNDLE}: {len(files)} files")


if __name__ == "__main__":
    main()
