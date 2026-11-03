# SPDX-License-Identifier: Apache-2.0
"""Launch a loopback server with an explicit source variant."""
import argparse
import os
from pathlib import Path
import shlex
import sys
from common import ROOT, gpu_snapshot, server_environment, verify_runtime, lock

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['stock', 'fix'], required=True)
    p.add_argument('--snapshot', choices=list(lock()['snapshots']), default='initial')
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, default=8300)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    command = [str(Path(sys.executable).parent / 'trtllm-serve'), str(Path(a.model).resolve()), '--host', '127.0.0.1', '--port', str(a.port), '--config', str(ROOT / lock()['snapshots'][a.snapshot].get('runtime_config', 'benchmarks/runtime.yaml')), '--generation-config', 'trtllm', '--no-telemetry']
    print(shlex.join(command), flush=True)
    if not a.execute:
        return
    verify_runtime(a.model)
    gpu_snapshot(require_idle=True)
    a.output.mkdir(parents=True, exist_ok=False)
    env = server_environment('baseline' if a.variant == 'stock' else 'candidate', a.snapshot, a.output)
    os.execvpe(command[0], command, env)

if __name__ == '__main__':
    main()
