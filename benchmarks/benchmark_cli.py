# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
from runner import ROOT, execute, plan
from common import read

def main(profile=False):
    p = argparse.ArgumentParser(description='Compare stock and GDN replay using frozen benchmark presets')
    p.add_argument('preset', choices=['c1', 'c8-short', 'c8-long', 'c8-cohort', 'c1-1024', 'c8-1024'])
    p.add_argument('--model', help='Existing local Qwen model directory')
    p.add_argument('--output', type=Path)
    p.add_argument('--port', type=int, default=8300)
    p.add_argument('--timeout-minutes', type=float, default=75)
    p.add_argument('--nsys', help='Installed Nsight Systems executable')
    p.add_argument('--execute', action='store_true', help='Launch the displayed experiment; otherwise print its scope only')
    a = p.parse_args()
    preset = read(ROOT / 'benchmarks/presets' / (a.preset + '.json'))
    config = plan(preset, profile=profile)
    print(json.dumps({k: v for k, v in config.items() if k != 'runs'}, indent=2), flush=True)
    if a.execute:
        if not a.model or not a.output:
            p.error('--execute requires --model and a fresh --output directory')
        execute(preset, a.output, str(Path(a.model).expanduser().resolve()), profile=profile, nsys=a.nsys, port=a.port, timeout_minutes=a.timeout_minutes)

if __name__ == '__main__':
    main()
