# SPDX-License-Identifier: Apache-2.0
"""Fetch the locked source revision into an ignored cache."""
import subprocess
from common import CACHE, check, lock

def fetch():
    source = CACHE / 'upstream'
    pinned = lock()['upstream']
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(['git', 'init', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'remote', 'add', 'origin', pinned['url']], check=True)
    url = subprocess.check_output(['git', '-C', str(source), 'remote', 'get-url', 'origin'], text=True).strip()
    check(url == pinned['url'], 'Cached upstream URL differs')
    found = subprocess.run(['git', '-C', str(source), 'cat-file', '-e', pinned['commit'] + '^{commit}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if found.returncode:
        subprocess.run(['git', '-C', str(source), '-c', 'fetch.recurseSubmodules=false', 'fetch', '--depth=1', '--filter=blob:none', 'origin', pinned['commit']], check=True, timeout=300)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', pinned['commit']], text=True).strip()
    check(actual == pinned['commit'], 'Fetched revision differs')
    print('Upstream source ready:', source)

if __name__ == '__main__':
    fetch()
