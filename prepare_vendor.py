"""Fetch the pinned Apache-2.0 upstream source and verify every file hash."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
COMMIT = "3c867e75ca22fd6eb01ac21e41fdc77f47b06ca8"
URL = f"https://codeload.github.com/google-deepmind/offpolicy_selection_eslb/zip/{COMMIT}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "vendor_manifest.json").read_text())
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    prefix = f"offpolicy_selection_eslb-{COMMIT}/"
    payloads = {}
    for rel, expected in manifest.items():
        name = str(Path(rel).relative_to("vendor/offpolicy_selection_eslb"))
        payload = (COMMIT + "\n").encode() if name == "UPSTREAM_COMMIT.txt" else archive.read(prefix + name)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        destination = args.destination / rel
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite modified source: {destination}")
        payloads[destination] = payload
    for destination, payload in payloads.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    print(f"Verified {len(payloads)} files from upstream commit {COMMIT}.")


if __name__ == "__main__":
    main()
