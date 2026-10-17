"""Verify public E9 package, frozen sources, and archived original run hashes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    count = 0
    for line in (ROOT / "RELEASE_MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert digest(ROOT / name) == expected, name
        count += 1
    for name, expected in json.loads((ROOT / "manifest.json").read_text())["sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in json.loads((ROOT / "results/source_hashes.json").read_text()).items():
        assert digest(ROOT / name) == expected, name
    print(f"Verified {count} public files, original run manifest, and frozen scientific sources.")


if __name__ == "__main__":
    main()
