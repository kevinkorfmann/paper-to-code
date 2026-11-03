"""Build checksum-pinned official ANGSD and HTSlib locally, without global install."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / '.cache/angsd-af'
REVISION = 'b3099ed756e7aff87d2d1dd4fd30cdf36998380b'
SOURCE = 'angsd-' + REVISION
BINARY = CACHE / 'build' / SOURCE / 'angsd'
RELEASES = {
    SOURCE: ('https://github.com/ANGSD/angsd/archive/' + REVISION + '.tar.gz',
             'e3da076b523e7d1fc32450019ff9d99f5cd2db722d260ce236bb775bc163fdc7'),
    'htslib-1.24': ('https://github.com/samtools/htslib/releases/download/1.24/htslib-1.24.tar.bz2',
                    '28a8de191381c7a97a35675ceac76fa1ea95e7b678d6a2e9d600a7874e4077de'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_version(binary=BINARY):
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    if 'angsd version: 0.941 (htslib: 1.24)' not in result.stdout + result.stderr:
        raise ValueError('Requires ANGSD 0.941 linked to HTSlib 1.24')
    return {'angsd': '0.941', 'htslib': '1.24'}


def install():
    downloads = CACHE / 'downloads'
    build = CACHE / 'build'
    downloads.mkdir(parents=True, exist_ok=True)
    build.mkdir(exist_ok=True)
    stamp = CACHE / 'build.json'
    hashes = {name: pair[1] for name, pair in RELEASES.items()}
    if stamp.exists() and BINARY.exists():
        record = json.loads(stamp.read_text())
        if record.get('archives') != hashes or record.get('binary_sha256') != digest(BINARY):
            raise ValueError('Cached build checksum mismatch')
        check_version()
        return BINARY
    for name, (url, sha) in RELEASES.items():
        archive = downloads / (name + '.tar')
        if not archive.exists():
            temporary = archive.with_suffix('.part')
            subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                            '--max-time', '120', url, '-o', str(temporary)],
                           check=True, timeout=130)
            if digest(temporary) != sha:
                raise ValueError('Official archive checksum mismatch: ' + name)
            temporary.replace(archive)
        if digest(archive) != sha:
            raise ValueError('Cached archive checksum mismatch: ' + name)
        with tarfile.open(archive) as source:
            if any(Path(item.name).parts[0] != name for item in source.getmembers()):
                raise ValueError('Unexpected archive root')
            source.extractall(build, filter='data')
    hts = build / 'htslib-1.24'
    commands = [(hts, ['./configure', '--disable-bz2', '--disable-lzma',
                       '--disable-libcurl', '--disable-s3', '--disable-gcs',
                       '--without-libdeflate']),
                (hts, ['make', '-j2', 'libhts.a']),
                (build / SOURCE, ['make', '-j2', 'angsd', 'HTSSRC=' + str(hts),
                                  'LIBS=' + str(hts / 'libhts.a') + ' -lz -lm -lpthread'])]
    with (CACHE / 'build.log').open('w') as log:
        for directory, command in commands:
            subprocess.run(command, cwd=directory, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=360)
    versions = check_version()
    stamp.write_text(json.dumps({'archives': hashes, 'binary_sha256': digest(BINARY),
                                'versions': versions}, indent=2) + '\n')
    return BINARY


if __name__ == '__main__':
    print(install())
