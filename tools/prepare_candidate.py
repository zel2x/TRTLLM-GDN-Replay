# SPDX-License-Identifier: Apache-2.0
"""Reconstruct only the modified Python modules; use installed CUDA binaries."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from common import CACHE, ROOT, check, lock, save, source_bytes, verify_inputs, verify_prepared

def prepare(snapshot):
    pinned = lock()
    check(snapshot in pinned['snapshots'], 'Unknown source snapshot')
    verify_inputs()
    destination = CACHE / 'prepared' / snapshot
    if destination.exists():
        return verify_prepared(snapshot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=snapshot + '-', dir=destination.parent))
    try:
        overlay = stage / 'overlay'
        overlay.mkdir()
        for name, expected in pinned['base_files'].items():
            data = source_bytes(name)
            check(hashlib.sha256(data).hexdigest() == expected, 'Upstream source mismatch: ' + name)
            file = overlay / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
        for patch in pinned['snapshots'][snapshot]['patches']:
            cmd = ['git', '--git-dir=' + str(CACHE / 'upstream/.git'), '--work-tree=' + str(overlay), 'apply']
            subprocess.run(cmd + ['--check', str(ROOT / patch)], cwd=overlay, check=True)
            subprocess.run(cmd + [str(ROOT / patch)], cwd=overlay, check=True)
        for name, expected in pinned['snapshots'][snapshot]['files'].items():
            data = (overlay / name).read_bytes()
            check(hashlib.sha256(data).hexdigest() == expected, 'Patched source mismatch: ' + name)
        save(stage / 'prepared.json', {'snapshot': snapshot, 'upstream': pinned['upstream'], 'files': pinned['snapshots'][snapshot]['files']})
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_prepared(snapshot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', choices=[*lock()['snapshots'], 'all'], default='all')
    args = parser.parse_args()
    for name in lock()['snapshots'] if args.snapshot == 'all' else [args.snapshot]:
        print('Prepared:', prepare(name))
