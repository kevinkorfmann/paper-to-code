"""Download a checksum-pinned official executable, without redistributing it."""

import hashlib
import io
from pathlib import Path
import platform
import tarfile
import urllib.request

URL = "https://dalexander.github.io/admixture/binaries/admixture_linux-1.4.0.tar.gz"
ARCHIVE_SHA256 = "5520c4f18525f49ab2c715d711b40f4a0818ef1485361a94ec8954d1364e6f19"
BINARY_SHA256 = "dbab7f695dc159bddd72a1137f11cb2888ca2939565e6123fbb80e001c4e613f"
VERSION = "1.4.0"
ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / ".cache" / "admixture-1.4.0" / "admixture"


def verify(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Dependency checksum mismatch; refusing to use these bytes")


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This reference executable requires Linux x86_64")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read()
    verify(archive, ARCHIVE_SHA256)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as packed:
        # Read exactly one member; never extract arbitrary paths or other files.
        binary = packed.extractfile("admixture_linux-1.4.0/admixture").read()
    verify(binary, BINARY_SHA256)
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    BINARY.write_bytes(binary)
    BINARY.chmod(0o755)
    print(f"Installed checksum-verified ADMIXTURE {VERSION}")


if __name__ == "__main__":
    main()
