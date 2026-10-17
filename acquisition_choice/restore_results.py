"""Restore the original E9 outcomes without overwriting changed local results."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    archive = ROOT / "results.zip"
    expected = dict(line.split("  ", 1)[::-1] for line in
                    (ROOT / "RELEASE_MANIFEST.sha256").read_text().splitlines())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected["results.zip"]
    with zipfile.ZipFile(archive) as source:
        entries = []
        for name in source.namelist():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "results":
                raise ValueError(f"Unexpected archive path: {name}")
            data = source.read(name)
            path = ROOT / relative
            if path.exists() and path.read_bytes() != data:
                raise FileExistsError(f"Refusing to replace modified result: {name}")
            entries.append((path, data))
        for path, data in entries:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    print(f"Verified and restored {len(entries)} result files.")


if __name__ == "__main__":
    main()
