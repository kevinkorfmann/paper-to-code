"""Install one checksum-pinned, unmodified official Linux x86-64 binary locally."""

import hashlib
import io
from pathlib import Path
import platform
import urllib.request
import zipfile

URL = "https://s3.amazonaws.com/plink2-assets/beta1/plink2_linux_x86_64_20261024.zip"
ARCHIVE_SHA256 = "b639901022d6b553d422eb571cfee8ff476b4f74babf258a81117b911ff45e41"
BINARY_SHA256 = "50ff05c332e8a4ff75f9bdbb9f513b0eab639e604c986ecd8b84b7f664253db2"
VERSION = "PLINK v2.0.0-b.1.2LM 64-bit Intel (24 Oct 2026)"
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / ".cache" / "plink2-20261024"
BINARY = BUILD / "plink2"


def verify(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Dependency checksum mismatch; refusing to use these bytes")


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This reference environment requires Linux x86_64")
    BUILD.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read()
    verify(archive, ARCHIVE_SHA256)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        binary = zipped.read("plink2")
        verify(binary, BINARY_SHA256)
        # Extract only named files, never arbitrary archive paths or other executables.
        BINARY.write_bytes(binary)
        (BUILD / "intel-simplified-software-license.txt").write_bytes(
            zipped.read("intel-simplified-software-license.txt"))
    BINARY.chmod(0o755)
    print(f"Installed checksum-verified {VERSION}")


if __name__ == "__main__":
    main()
