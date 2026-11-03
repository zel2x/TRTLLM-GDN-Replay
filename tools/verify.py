# SPDX-License-Identifier: Apache-2.0
import argparse
from common import lock, verify_inputs, verify_prepared, verify_runtime

def main():
    p = argparse.ArgumentParser(description='Verify packaged source, benchmark inputs and optionally the inference environment')
    p.add_argument('--model')
    args = p.parse_args()
    verify_inputs()
    for snapshot in lock()['snapshots']:
        verify_prepared(snapshot)
    if args.model:
        verify_runtime(args.model)
    print('Source and benchmark checks passed' + ('; runtime and model checks passed' if args.model else ''))

if __name__ == '__main__':
    main()
