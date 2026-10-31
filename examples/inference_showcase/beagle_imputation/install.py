"""Fetch official dependencies into an ignored cache; no system installation."""

import hashlib
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / '.cache' / 'beagle-imputation'
BEAGLE_VERSION = '5.5 (27Feb25.75f)'
JAVA_VERSION = '17.0.21+9'
JAR_NAME = 'beagle.27Feb25.75f.jar'
JAR_URL = 'https://faculty.washington.edu/browning/beagle/' + JAR_NAME
JAR_SHA256 = '7319f4af9638be05c18dcc1bfb8fb41a58a09293507ebf0d54617d0e40df5a70'
JAR = CACHE / JAR_NAME
RELEASE = 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.21%2B9/'
PACKAGES = {
    ('Darwin', 'arm64'): ('OpenJDK17U-jre_aarch64_mac_hotspot_17.0.21_9.tar.gz',
        'a5ab5bacd6e719d5a5570580ed14dceb2ed532208a39af927906b9742da1745e',
        'jdk-17.0.21+9-jre/Contents/Home'),
    ('Linux', 'x86_64'): ('OpenJDK17U-jre_x64_linux_hotspot_17.0.21_9.tar.gz',
        '9f41bf88c7900c7097444a1bdeb84146ae692bc08842490ab098063c74bf9a8a',
        'jdk-17.0.21+9-jre'),
}


def verify(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Dependency checksum mismatch')


def package(system=None, machine=None):
    key = (system or platform.system(), machine or platform.machine())
    if key not in PACKAGES:
        raise RuntimeError('Pinned runtimes support macOS arm64 and Linux x86_64 only')
    return PACKAGES[key]


def fetch(url, destination, checksum):
    if destination.exists():
        verify(destination.read_bytes(), checksum)
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    verify(data, checksum)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def runtime():
    _, _, relative = package()
    java = CACHE / relative / 'bin/java'
    if not java.is_file() or not JAR.is_file():
        raise RuntimeError('First run python -m examples.inference_showcase.beagle_imputation.install')
    verify(JAR.read_bytes(), JAR_SHA256)
    result = subprocess.run([str(java), '-version'], capture_output=True, text=True, check=True, timeout=15)
    if 'Temurin-17.0.21+9' not in result.stderr:
        raise RuntimeError('Expected pinned Temurin 17.0.21+9')
    return java


def main():
    name, checksum, relative = package()
    archive = CACHE / name
    fetch(RELEASE + name, archive, checksum)
    if not (CACHE / relative / 'bin/java').exists():
        with tarfile.open(archive, 'r:gz') as packed:
            # The checksum identifies the complete official archive; data filtering
            # also rejects absolute/escaping members and unsafe link destinations.
            packed.extractall(CACHE, filter='data')
    fetch(JAR_URL, JAR, JAR_SHA256)
    runtime()
    print(f'Installed Beagle {BEAGLE_VERSION}; Temurin {JAVA_VERSION} in the project cache')


if __name__ == '__main__':
    main()
