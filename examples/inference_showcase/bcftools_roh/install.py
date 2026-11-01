"""Build checksum-pinned official releases in ignored .cache; never install globally."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT/'.cache/bcftools-roh'
BINARY = CACHE/'build/bcftools-1.23.1/bcftools'
RELEASES = {
    'bcftools-1.23.1': ('https://github.com/samtools/bcftools/releases/download/1.23.1/bcftools-1.23.1.tar.bz2',
                       '01899a46f9420cdc1385d52fcfc84cce2806f9c996b787081a90d7dfc85eafa3'),
    'htslib-1.24': ('https://github.com/samtools/htslib/releases/download/1.24/htslib-1.24.tar.bz2',
                    '28a8de191381c7a97a35675ceac76fa1ea95e7b678d6a2e9d600a7874e4077de'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_version(binary):
    result = subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True,timeout=10)
    lines = result.stdout.splitlines()
    if lines[:2] != ['bcftools 1.23.1','Using htslib 1.24']:
        raise ValueError('Requires BCFtools 1.23.1 linked to HTSlib 1.24')
    return {'bcftools':'1.23.1','htslib':'1.24'}


def install():
    downloads=CACHE/'downloads';build=CACHE/'build'
    downloads.mkdir(parents=True,exist_ok=True);build.mkdir(exist_ok=True)
    stamp=CACHE/'build.json'
    if stamp.exists() and BINARY.exists():
        recorded=json.loads(stamp.read_text())
        if recorded.get('archives') != {k:v[1] for k,v in RELEASES.items()} or recorded.get('binary_sha256') != digest(BINARY):
            raise ValueError('Existing cached build does not match its recorded checksums')
        check_version(BINARY)
        return BINARY
    for name,(url,sha) in RELEASES.items():
        archive=downloads/(name+'.tar.bz2')
        if not archive.exists():
            temporary=archive.with_suffix('.part')
            subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',url,'-o',str(temporary)],check=True,timeout=70)
            if digest(temporary)!=sha:raise ValueError('Official archive checksum mismatch: '+name)
            temporary.replace(archive)
        if digest(archive)!=sha:raise ValueError('Cached archive checksum mismatch: '+name)
        with tarfile.open(archive) as source:
            if any(Path(member.name).parts[0] != name for member in source.getmembers()):
                raise ValueError('Unexpected archive root')
            source.extractall(build,filter='data')
    hts=build/'htslib-1.24';bcf=build/'bcftools-1.23.1'
    commands=[(hts,['./configure','--disable-bz2','--disable-lzma','--disable-libcurl',
                    '--disable-s3','--disable-gcs','--without-libdeflate']),
              (hts,['make','-j2','libhts.a']),
              (bcf,['./configure','--with-htslib='+str(hts),'--disable-libgsl',
                    '--disable-perl-filters','--disable-bcftools-plugins']),
              (bcf,['make','-j2','bcftools'])]
    with (CACHE/'build.log').open('w') as log:
        for directory,command in commands:
            subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    versions=check_version(BINARY)
    stamp.write_text(json.dumps({'archives':{k:v[1] for k,v in RELEASES.items()},
                                'binary_sha256':digest(BINARY),'versions':versions,
                                'scope':'Local CLI build without GSL/plugins/network codecs; no global installation'},indent=2)+'\n')
    return BINARY


if __name__=='__main__':
    print(install())
