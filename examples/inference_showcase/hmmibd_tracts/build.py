"""Fetch, verify, and compile an unmodified external GPL-3.0 dependency."""

import hashlib
from pathlib import Path
import subprocess
import urllib.request

COMMIT = "c553843db8a7c070c50bdd96623e1f8280a8009d"
SOURCE_SHA256 = "03d08b190ceedefa1e834ae1525c1cd5ff778b9818777ed8e1e4e3a58defd922"
LICENSE_SHA256 = "589ed823e9a84c56feb95ac58e7cf384626b9cbf4fda2a907bc36e103de1bad2"
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / ".cache" / "hmmibd"
BINARY = BUILD / "hmmIBD"


def verify(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("External dependency checksum mismatch; refusing to compile")


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, checksum in [("hmmIBD.c", SOURCE_SHA256), ("LICENSE", LICENSE_SHA256)]:
        path = BUILD / name
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/glipsnort/hmmIBD/{COMMIT}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        verify(data, checksum)
        path.write_bytes(data)
    subprocess.run(["cc", "-O3", "-Wall", str(BUILD / "hmmIBD.c"), "-lm", "-o", str(BINARY)],
                   check=True, timeout=60)
    print(f"Built checksum-verified hmmIBD {COMMIT}")


if __name__ == "__main__":
    main()
