# SPDX-License-Identifier: Apache-2.0
"""Shared, CPU-only source and environment checks."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def lock():
    return read(ROOT / 'upstream.lock.json')

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def verify_inputs():
    for name, expected in lock()['input_files'].items():
        check(digest(ROOT / name) == expected, 'Input hash mismatch: ' + name)

def source_bytes(name):
    pinned = lock()['upstream']
    return subprocess.check_output(['git', '-C', str(CACHE / 'upstream'), 'show', pinned['commit'] + ':' + name])

def verify_prepared(snapshot):
    verify_inputs()
    prepared = CACHE / 'prepared' / snapshot
    receipt = read(prepared / 'prepared.json')
    check(receipt['snapshot'] == snapshot and receipt['upstream'] == lock()['upstream'], 'Prepared source identity mismatch')
    for name, expected in lock()['snapshots'][snapshot]['files'].items():
        check(digest(prepared / 'overlay' / name) == expected, 'Prepared file mismatch: ' + name)
    return prepared

def verify_runtime(model):
    pinned = lock()
    check(sys.platform == 'linux', 'Run inference in Linux/WSL using the matching CUDA environment')
    check(list(sys.version_info[:2]) == pinned['python'], 'Python does not match upstream.lock.json')
    actual = {name: metadata.version(name) for name in pinned['packages']}
    check(actual == pinned['packages'], 'Installed packages do not match upstream.lock.json')
    trt = metadata.distribution('tensorrt_llm')
    for name, expected in pinned['base_files'].items():
        check(digest(trt.locate_file(name)) == expected, 'Installed TRT-LLM source differs: ' + name)
    fi = metadata.distribution('flashinfer-python')
    check(digest(fi.locate_file(pinned['flashinfer']['file'])) == pinned['flashinfer']['sha256'], 'Installed FlashInfer source differs')
    model = Path(model).expanduser().resolve()
    check(model.is_dir() and (model / 'config.json').is_file(), 'Supply an existing local model directory')
    revision = pinned['model']['revision']
    files = [model / 'config.json'] + list(model.glob('*.safetensors'))
    check(len(files) > 1, 'Model weights are missing')
    for path in files:
        marker = model / '.cache/huggingface/download' / (path.name + '.metadata')
        check(revision in model.parts or (marker.is_file() and marker.read_text().splitlines()[0] == revision), 'Model revision metadata differs: ' + path.name)
    check(digest(model / 'tokenizer.json') == pinned['model']['tokenizer_sha256'], 'Tokenizer differs from frozen fixtures')
    for name, expected in pinned['benchmark_files'].items():
        check(digest(trt.locate_file(name)) == expected, 'Official benchmark source differs: ' + name)
    cuda = cuda_home()
    banner = subprocess.check_output([str(cuda / 'bin/nvcc'), '--version'], text=True, timeout=15)
    check('release ' + pinned['cuda_toolkit_release'] + ',' in banner, 'CUDA toolkit differs from lock file')
    serve = Path(sys.executable).parent / 'trtllm-serve'
    check(serve.is_file(), 'trtllm-serve is missing from the selected environment')
    return {'python': sys.executable, 'packages': actual, 'model': str(model), 'serve': str(serve), 'cuda_home': str(cuda)}

def gpu_snapshot(require_idle=False):
    binary = '/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').is_file() else 'nvidia-smi'
    def query(args):
        return subprocess.check_output([binary, *args], text=True, timeout=15).strip()
    apps = query(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    if require_idle:
        architecture = query(['--query-gpu=compute_cap', '--format=csv,noheader'])
        check(architecture == '12.0', 'This benchmark requires the single SM120 target')
        check(not apps, 'GPU has active compute processes; leave them untouched and retry later')
    return {'apps': apps, 'gpu': query(['--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader'])}

def cuda_home():
    if os.environ.get('CUDA_HOME'):
        home = Path(os.environ['CUDA_HOME']).expanduser().resolve()
    else:
        compiler = shutil.which('nvcc')
        check(bool(compiler), 'Activate the CUDA toolkit or set CUDA_HOME')
        home = Path(compiler).resolve().parents[1]
    check((home / 'bin/nvcc').is_file(), 'CUDA_HOME does not contain bin/nvcc')
    return home

def runtime_environment():
    env = os.environ.copy()
    for key in env:
        check(not key.startswith('E2E_') and key not in ['TRTLLM_LOW_M_GEMM_BACKEND', 'FORCE_DETERMINISTIC'], 'Remove inherited optimization control: ' + key)
    env.pop('OPENAI_API_KEY', None)
    cuda = cuda_home()
    env.update(lock()['environment'])
    env.update(CUDA_HOME=str(cuda), PATH=os.pathsep.join([str(Path(sys.executable).parent), str(cuda / 'bin'), '/usr/lib/wsl/lib', env.get('PATH', '')]))
    return env

def server_environment(variant, snapshot, directory, *, stats=False, profile=False):
    check(variant in ['baseline', 'candidate'], 'Invalid variant')
    prepared = verify_prepared(snapshot)
    env = runtime_environment()
    check(not any(k.startswith('STUDY_') or k.startswith('GDN_REPLAY_') for k in env), 'Remove inherited experiment variables before launching')
    check(not env.get('PYTHONPATH'), 'Launch from a clean environment without an inherited PYTHONPATH')
    check(not any(k.startswith('NSYS_') or k in ['CUDA_INJECTION64_PATH', 'NVTX_INJECTION64_PATH'] for k in env), 'Remove inherited profiler injection variables')
    check(not any(word in env.get('LD_PRELOAD', '').lower() for word in ['nsys', 'nsight']), 'Remove inherited profiler LD_PRELOAD')
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / 'benchmarks/hooks'), str(ROOT / 'benchmarks/support')]),
               GDN_REPLAY_PREPARED=str(prepared), STUDY_RUN_DIR=str(Path(directory).resolve()),
               STUDY_VARIANT=variant, PYTHONFAULTHANDLER='1')
    if stats or profile:
        env['STUDY_COLLECT_STATS'] = '1'
    if profile:
        env.update(STUDY_PROFILE='1', GDN_STATS_CAPACITY='32768')
    return env


def measurement_preset(preset, profile):
    result = dict(preset)
    if profile:
        result['collect_stats'] = True
        for key in ['warmup', 'measured', 'warmup_fixture', 'measured_fixture']:
            if key in preset['profile']:
                result[key] = preset['profile'][key]
    return result
