# SPDX-License-Identifier: Apache-2.0
"""Bounded stock/fix process pairs, with isolated workers and owned-process cleanup."""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'benchmarks/support'))
from common import check, digest, gpu_snapshot, read, save, server_environment, verify_inputs, verify_prepared, verify_runtime, runtime_environment, measurement_preset
from analyze import analyze_run

def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None

def stop_owned(process, ticks, timeout):
    if process.poll() is None and identity(process.pid) == ticks:
        os.kill(process.pid, signal.SIGTERM)
    process.wait(timeout=timeout)

def drain(directory):
    observations = []
    end = time.monotonic() + 60
    while time.monotonic() < end:
        row = gpu_snapshot()
        observations.append(row)
        if not row['apps']:
            save(directory / 'gpu_drain.json', observations)
            return
        time.sleep(.5)
    save(directory / 'gpu_drain.json', observations)
    raise RuntimeError('GPU processes remain after owned-server shutdown')

def choose_replay_plans(snapshots):
    return max(snapshots, key=lambda v: (sum(y['replays'] for y in v), len(v)))

def validate_replay_counts(summary, preset):
    check(summary['GDN_fallbacks'] == 0, 'Unexpected replay fallback')
    check(summary['GDN_replays'] > 0 or preset.get('allow_zero_replay', False) or preset['name'] == 'c8-cohort', 'Replay not reached')

def collect_audit(directory, variant, preset, *, profile):
    rows = [json.loads(line) for f in directory.glob('timing_audit.*.jsonl') for line in f.read_text().splitlines()]
    caps = [x for x in rows if x.get('event') == 'capacity']
    check(bool(caps), 'Missing runtime/cache audit')
    capacity = caps[-1]
    check(capacity['logical_max_tokens'] == 24576 and capacity['physical_capacity_diagnostic']['maxNumTokens'] == 19776, 'KV capacity differs from benchmark contract')
    check(any(x.get('event') == 'allocation' and x['dtype'] == 'DataType.BF16' for x in rows), 'Missing BF16 cache evidence')
    check(any(x.get('event') == 'FI_control_installed' and x['all_aliases_asserted'] for x in rows), 'Common FI control was not installed')
    summary = {'logical_KV': 24576, 'physical_KV': 19776, 'allocation_dtype': 'BF16'}
    if variant == 'candidate':
        snapshots = [x['live_plans'] for x in rows if x.get('event') == 'snapshot']
        check(bool(snapshots), 'Missing candidate replay-plan audit')
        plans = choose_replay_plans(snapshots)
        sizes = preset.get('capture_sizes', [128])
        check(len(plans) == 18 * len(sizes) and all(x['owner_alive'] for x in plans), 'Candidate replay plans/ownership differ')
        summary.update(GDN_replays=sum(x['replays'] for x in plans), GDN_fallbacks=sum(x['fallbacks'] for x in plans))
        if preset.get('require_replay_tokens'):
            by_size = {}
            for size in sizes:
                group = [x for x in plans if x['counts'] == [1, size, size, 1, size]]
                check(len(group) == 18, 'Missing captured GDN layers for size ' + str(size))
                by_size[str(size)] = {'layers': len(group), 'replays': sum(x['replays'] for x in group), 'fallbacks': sum(x['fallbacks'] for x in group)}
            summary['GDN_by_capture_size'] = by_size
            required = str(preset['require_replay_tokens'])
            check(by_size[required]['replays'] > 0, 'Required 1024-token GDN replay was not used')
            minimum = 18 * (preset['warmup'] + preset['measured']) // preset['concurrency']
            check(by_size[required]['replays'] >= minimum, 'Missing eligible 1024-token replay coverage')
            if preset['concurrency'] == 1:
                check(by_size[required]['replays'] == minimum and summary['GDN_fallbacks'] == 0, 'C1 replay/fallback coverage differs')
        else:
            validate_replay_counts(summary, preset)
        if preset['name'] == 'c1':
            check(summary['GDN_replays'] == 18 * (preset['warmup'] + preset['measured']), 'C1 replay coverage differs')
    if preset['collect_stats'] or profile:
        stats = [x for x in rows if x.get('event') == 'control_iteration_rows']
        check(bool(stats), 'Missing executor iteration audit')
        stats = stats[-1]
        check(not stats['overflow'] and not stats['collector_error'], 'Executor statistics overflow/error')
        check(stats['thread_stopped'] or identity(read(directory / 'server_identity.json')['pid']) is None, 'Executor collector is still active')
        traffic = read(directory / 'client/client_receipt.json')['traffic'][1]
        offset = read(directory / 'manifest.json')['UTC_offset_seconds']
        selected = []
        for row in stats['rows']:
            stamp = datetime.datetime.strptime(row['timestamp'], '%m-%d-%Y %H:%M:%S.%f')
            unix = stamp.replace(tzinfo=datetime.timezone.utc).timestamp() - offset
            if traffic['start_unix'] <= unix <= traffic['end_unix']:
                selected.append(row)
        check(bool(selected), 'No executor statistics inside measurement')
        histogram = {}
        for row in selected:
            key = f"{row['context']}prefill+{row['generation']}decode"
            histogram[key] = histogram.get(key, 0) + 1
        summary['iterations'] = {'count': len(selected), 'composition': histogram, 'max_generation_requests': max(x['generation'] for x in selected), 'mixed': sum(x['context'] > 0 and x['generation'] > 0 for x in selected)}
        if preset['name'] == 'c1':
            check(len(selected) == preset['measured'] * 128 and summary['iterations']['max_generation_requests'] == 1 and summary['iterations']['mixed'] == 0, 'C1 execution shape differs')
        elif preset['name'] == 'c1-1024':
            check(summary['iterations']['max_generation_requests'] == 1 and summary['iterations']['mixed'] == 0, 'C1 execution shape differs')
        else:
            check(summary['iterations']['max_generation_requests'] >= 8, 'Actual batched execution did not reach eight sequences')
    if preset['name'] == 'c8-cohort':
        eligible = sum(row['context'] == 1 and row['generation'] == 0 for row in selected)
        summary['measured_eligible_prefill_forwards'] = eligible
        if variant == 'candidate':
            check(summary['GDN_replays'] >= eligible * 18, 'Missing replay for an eligible synchronized prefill')
    save(directory / 'execution.json', summary)
    return summary

def one_run(config, row, output, model, port, nsys):
    preset, profile = config['preset'], config['profiled']
    runtime = verify_runtime(model)
    before = gpu_snapshot(require_idle=True)
    D = output / row['label']
    D.mkdir()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    env = server_environment(row['variant'], preset['snapshot'], D, stats=preset['collect_stats'], profile=profile)
    serve = str(Path(sys.executable).parent / 'trtllm-serve')
    tail = [model, '--host', '127.0.0.1', '--port', str(port), '--config', str(ROOT / preset.get('runtime_config', 'benchmarks/runtime.yaml')), '--generation-config', 'trtllm', '--no-telemetry']
    if profile:
        command = [nsys, 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=process-tree', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--cuda-graph-trace=node', '--duration=' + str(preset['profile']['capture_seconds']), '--kill=none', '--wait=primary', '--stats=false', '-o', str(D / 'capture'), sys.executable, str(ROOT / 'benchmarks/support/server_entry.py'), *tail]
    else:
        command = [serve, *tail]
    manifest = {'command': command, 'variant': row['variant'], 'snapshot': preset['snapshot'], 'profiled': profile, 'GPU_before': before, 'UTC_offset_seconds': datetime.datetime.now().astimezone().utcoffset().total_seconds(), 'runtime': runtime}
    save(D / 'manifest.json', manifest)
    check_remaining = lambda: check(time.monotonic() < config['_deadline'], 'Benchmark deadline reached')
    process = None
    owner = None
    stop = threading.Event()
    telemetry = None
    def monitor():
        with (D / 'telemetry.jsonl').open('w') as stream:
            while not stop.wait(5):
                try:
                    stream.write(json.dumps({'unix': time.time(), **gpu_snapshot()}) + '\n')
                except Exception as exc:
                    stream.write(json.dumps({'error': type(exc).__name__}) + '\n')
                stream.flush()
    try:
        with (D / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            ticks = identity(process.pid)
            if not profile:
                owner = {'pid': process.pid, 'start_ticks': ticks}
                save(D / 'server_identity.json', owner)
            row.update(state='starting', pid=process.pid, start_ticks=ticks)
            save(output / 'run.json', {k: v for k, v in config.items() if not k.startswith('_')})
            telemetry = threading.Thread(target=monitor, daemon=True)
            telemetry.start()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            ready_deadline = min(config['_deadline'], time.monotonic() + 300)
            while time.monotonic() < ready_deadline:
                check(process.poll() is None, 'Server/profiler exited before readiness; inspect server.log')
                if profile and (D / 'server_identity.json').is_file():
                    owner = read(D / 'server_identity.json')
                if owner:
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                time.sleep(1)
            else:
                raise RuntimeError('Server readiness timeout')
            check(identity(owner['pid']) == owner['start_ticks'], 'Server process identity changed')
            row['state'] = 'ready'
            save(output / 'run.json', {k: v for k, v in config.items() if not k.startswith('_')})
            client_dir = D / 'client'
            client_dir.mkdir()
            client_env = runtime_environment()
            client_env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'benchmarks/support'), STUDY_RUN_DIR=str(client_dir), STUDY_VARIANT='disabled')
            client = [sys.executable, str(ROOT / 'benchmarks/client.py'), '--preset', preset['name'], '--model', model, '--port', str(port), '--output', str(client_dir)] + (['--profile'] if profile else [])
            manifest['client_command'] = client
            save(D / 'manifest.json', manifest)
            check_remaining()
            with (client_dir / 'stdout.log').open('w') as log:
                subprocess.run(client, cwd=ROOT, env=client_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=min(600, max(1, config['_deadline'] - time.monotonic())), check=True)
    finally:
        try:
            if process is not None:
                if profile:
                    if owner is None and (D / 'server_identity.json').is_file():
                        owner = read(D / 'server_identity.json')
                    if owner and identity(owner['pid']) == owner['start_ticks']:
                        os.kill(owner['pid'], signal.SIGTERM)
                    elif owner is None and process.poll() is None:
                        stop_owned(process, ticks, 90)
                    process.wait(timeout=90)
                else:
                    stop_owned(process, ticks, 90)
                manifest['returncode'] = process.returncode
        finally:
            stop.set()
            if telemetry:
                telemetry.join(timeout=20)
            save(D / 'manifest.json', manifest)
    drain(D)
    if profile:
        check(process.returncode in [0, 143], 'Nsight did not finalize normally')
        report = D / 'capture.nsys-rep'
        check(report.is_file() and report.stat().st_size > 0, 'Nsight report missing')
        database = D / 'capture.sqlite'
        with (D / 'export.log').open('w') as log:
            subprocess.run([nsys, 'export', '--type=sqlite', '--output=' + str(database), str(report)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        from coverage_validator import read_nvtx, validate_coverage
        sidecars = []
        for path in D.rglob('pipeline.*.json'):
            payload = read(path)
            check(not payload['overflow'], 'Diagnostic buffer overflow')
            sidecars.extend(payload['events'])
        coverage = validate_coverage(read_nvtx(database), sidecars, preset['measured'], row['variant'], require_replay=not (preset.get('allow_zero_replay', False) or preset['name'] == 'c8-cohort'), required_replay_tokens=preset.get('require_replay_tokens'))
        save(D / 'coverage.json', coverage)
        manifest['report'] = {'bytes': report.stat().st_size, 'sha256': digest(report)}
    row['execution'] = collect_audit(D, row['variant'], preset, profile=profile)
    row['state'] = 'complete'
    save(D / 'manifest.json', manifest)
    return read(D / 'client/measured.json')

def plan(preset, *, profile):
    preset = measurement_preset(preset, profile)
    pairs = 1 if profile else preset['pairs']
    runs = []
    for pair in range(pairs):
        order = ['candidate', 'baseline'] if (pair % 2 == 1 or (profile and preset['name'] == 'c8-long')) else ['baseline', 'candidate']
        for variant in order:
            runs.append({'pair': pair, 'variant': variant, 'label': f'pair-{pair + 1}-' + ('stock' if variant == 'baseline' else 'fix'), 'state': 'planned'})
    return {'preset': preset, 'profiled': profile, 'servers': len(runs), 'clients': len(runs), 'official_calls': len(runs) * 2, 'requests': len(runs) * (preset['warmup'] + preset['measured']), 'runs': runs}

def execute(preset, output, model, *, profile, nsys, port, timeout_minutes):
    preset = measurement_preset(preset, profile)
    verify_inputs()
    verify_prepared(preset['snapshot'])
    verify_runtime(model)
    if profile:
        check(bool(nsys) and shutil.which(nsys) is not None, 'Supply an installed Nsight executable using --nsys')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = plan(preset, profile=profile)
    config.update(state='running', _deadline=time.monotonic() + timeout_minutes * 60)
    save(output / 'run.json', {k: v for k, v in config.items() if not k.startswith('_')})
    try:
        for i, row in enumerate(config['runs']):
            check(time.monotonic() < config['_deadline'], 'Benchmark deadline reached')
            print('Running', row['label'], flush=True)
            one_run(config, row, output, model, port + i, nsys)
            save(output / 'run.json', {k: v for k, v in config.items() if not k.startswith('_')})
        for pair in range(1 if profile else preset['pairs']):
            arms = [x for x in config['runs'] if x['pair'] == pair]
            files = {x['variant']: output / x['label'] / 'client' for x in arms}
            agreement = {}
            for phase in ['warmup', 'measured']:
                stock = read(files['baseline'] / (phase + '.json'))['generated_texts']
                fix = read(files['candidate'] / (phase + '.json'))['generated_texts']
                agreement[phase] = {'equal': sum(a == b for a, b in zip(stock, fix)), 'requests': len(stock)}
                if preset['name'] == 'c1':
                    check(stock == fix, 'C1 greedy-output agreement failed')
            save(output / f'output-agreement-{pair + 1}.json', agreement)
        config['state'] = 'complete'
        save(output / 'run.json', {k: v for k, v in config.items() if not k.startswith('_')})
        analysis = analyze_run(output)
        save(output / 'analysis.json', analysis)
        return analysis
    except BaseException as exc:
        config.update(state='stopped', error=repr(exc))
        save(output / 'run.json', {k: v for k, v in config.items() if not k.startswith('_')})
        raise
