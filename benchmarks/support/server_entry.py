# SPDX-License-Identifier: Apache-2.0
"""Record the owned server identity before Nsight's child exec."""
import json
import os
from pathlib import Path
import sys
D = Path(os.environ['STUDY_RUN_DIR'])
(D / 'server_identity.json').write_text(json.dumps({'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21]}) + '\n')
serve = Path(sys.executable).parent / 'trtllm-serve'
os.execv(str(serve), [str(serve), *sys.argv[1:]])
