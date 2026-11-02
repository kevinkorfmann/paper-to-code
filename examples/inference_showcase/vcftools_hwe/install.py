"""Build an official checksum-pinned CLI in ignored .cache, never globally."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / '.cache/vcftools-hwe'
VERSION = '0.1.17'
URL = 'https://github.com/vcftools/vcftools/releases/download/v0.1.17/vcftools-0.1.17.tar.gz'
SHA256 = 'b9e0e1c3e86533178edb35e02c6c4de9764324ea0973bebfbb747018c2d2a42c'
BINARY = CACHE / 'build/vcftools-0.1.17/src/cpp/vcftools'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_version(binary=BINARY):
    result = subprocess.run([str(binary), '--version'], check=True,
                            capture_output=True, text=True, timeout=10)
    if (result.stdout + result.stderr).strip() != 'VCFtools (0.1.17)':
        raise ValueError('This example requires VCFtools 0.1.17')
    return VERSION


def install():
    downloads = CACHE / 'downloads'
    build = CACHE / 'build'
    downloads.mkdir(parents=True, exist_ok=True)
    build.mkdir(exist_ok=True)
    stamp = CACHE / 'build.json'
    if stamp.exists() and BINARY.exists():
        record = json.loads(stamp.read_text())
        if record.get('archive_sha256') != SHA256 or record.get('binary_sha256') != digest(BINARY):
            raise ValueError('Cached build does not match recorded checksums')
        check_version()
        return BINARY
    archive = downloads / 'vcftools-0.1.17.tar.gz'
    if not archive.exists():
        part = downloads / 'download.part'
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                        '--max-time', '60', URL, '-o', str(part)], check=True, timeout=70)
        if digest(part) != SHA256:
            raise ValueError('Downloaded official archive checksum mismatch')
        part.replace(archive)
    if digest(archive) != SHA256:
        raise ValueError('Cached official archive checksum mismatch')
    with tarfile.open(archive) as source:
        if any(Path(member.name).parts[0] != 'vcftools-0.1.17'
               for member in source.getmembers()):
            raise ValueError('Unexpected archive root')
        source.extractall(build, filter='data')
    source_dir = build / 'vcftools-0.1.17'
    with (CACHE / 'build.log').open('w') as log:
        for command in (['./configure'], ['make', '-C', 'src/cpp', '-j2']):
            subprocess.run(command, cwd=source_dir, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=240)
    check_version()
    stamp.write_text(json.dumps({'version': VERSION, 'archive_sha256': SHA256,
                                'binary_sha256': digest(BINARY)}, indent=2) + '\n')
    return BINARY


if __name__ == '__main__':
    print(install())
