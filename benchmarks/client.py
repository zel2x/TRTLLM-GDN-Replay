# SPDX-License-Identifier: Apache-2.0
"""Use the official HTTP request implementation and metrics without changing them."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'benchmarks/support'))
from common import check, read, save, measurement_preset
from sampling_audit import describe, expected_records, validate_sampled, validate_result

def official_parser(official):
    tree = ast.parse(inspect.getsource(official))
    entry = next(n for n in tree.body if isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))
    prefix = []
    for node in entry.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
        prefix.append(node)
    namespace = dict(vars(official))
    exec(compile(ast.Module(body=prefix, type_ignores=[]), official.__file__, 'exec'), namespace)
    return namespace['parser']

def install_profile_tags(official):
    import aiohttp
    import diag_probe as probe
    from tensorrt_llm.serve.scripts import backend_request_func as backend
    original_request = official.ASYNC_REQUEST_FUNCS['openai']
    counter = [0]
    async def tagged(*args, **kwargs):
        counter[0] += 1
        tag = 'diag-' + str(counter[0])
        token = probe.TAG.set(tag)
        probe.emit('client_submit', tag=tag)
        try:
            return await original_request(*args, **kwargs)
        finally:
            probe.emit('client_done', tag=tag)
            probe.TAG.reset(token)
    official.ASYNC_REQUEST_FUNCS['openai'] = tagged
    original_http = aiohttp.ClientSession._request
    async def request(self, method, url, *args, **kwargs):
        if str(url).endswith('/v1/completions'):
            headers = dict(kwargs.get('headers') or {})
            headers['X-Study-Request-ID'] = probe.TAG.get() or ''
            kwargs['headers'] = headers
        return await original_http(self, method, url, *args, **kwargs)
    aiohttp.ClientSession._request = request
    original_sse = backend._iter_sse_data
    async def stream(content):
        async for chunk in original_sse(content):
            payload = json.loads(chunk)
            if payload.get('choices'):
                tag = probe.TAG.get()
                probe.emit('client_receive', tag=tag, seq=probe.next_sequence('client_sse', tag), response_id=str(payload.get('id', '')))
            yield chunk
    backend._iter_sse_data = stream
    return counter

def profile_control(port, endpoint, payload):
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/' + endpoint, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    before = time.time()
    with opener.open(request, timeout=40) as response:
        check(response.status == 200, 'Worker profile control failed')
        body = json.loads(response.read())
    return {'endpoint': endpoint, 'start_unix': before, 'end_unix': time.time(), 'body': body}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    preset = measurement_preset(read(ROOT / 'benchmarks/presets' / (a.preset + '.json')), a.profile)
    D = a.output
    check(D.is_dir() and not (D / 'client_receipt.json').exists(), 'Use a fresh client directory')
    from tensorrt_llm.serve.scripts import benchmark_serving as official
    parser = official_parser(official)
    concurrency = preset['concurrency']
    common = ['--backend', 'openai', '--base-url', f'http://127.0.0.1:{a.port}', '--endpoint', '/v1/completions', '--model', a.model, '--tokenizer', a.model, '--dataset-name', 'trtllm_custom', '--max-concurrency', str(concurrency), '--request-rate', preset['request_rate'], '--ignore-eos', '--temperature', '0', '--seed', str(preset['seed']), '--no-test-input', '--disable-tqdm', '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,95,99', '--save-result', '--save-detailed', '--result-dir', str(D)]
    configs = []
    for phase, count in [('warmup', preset['warmup']), ('measured', preset['measured'])]:
        fixture = ROOT / 'benchmarks/fixtures' / preset[phase + '_fixture']
        configs.append(parser.parse_args(common + ['--dataset-path', str(fixture), '--num-prompts', str(count), '--result-filename', phase + '.json']))
    traffic = []
    original = official.benchmark
    async def audited(*args, **kwargs):
        phase = 'warmup' if not traffic else 'measured'
        sampled = [describe(request) for request in kwargs['input_requests']]
        validate_sampled(sampled, expected_records(ROOT / 'benchmarks/fixtures' / preset[phase + '_fixture'], len(sampled)))
        row = {'phase': phase, 'start_unix': time.time(), 'request_count': len(sampled), 'sampled_requests': sampled}
        traffic.append(row)
        save(D / 'call_boundaries.json', traffic)
        try:
            return await original(*args, **kwargs)
        finally:
            row['end_unix'] = time.time()
            save(D / 'call_boundaries.json', traffic)
    official.benchmark = audited
    counter = install_profile_tags(official) if a.profile else None
    finish_arrivals = None
    if preset['arrival'] == 'synchronized_cohorts':
        from cohort_arrivals import install
        finish_arrivals = install(official, D, size=preset['cohort_size'], max_spread_ms=preset['max_client_start_spread_ms'], expected_counts=(preset['warmup'], preset['measured']))
    imported = time.time()
    official.main(configs[0])
    if a.profile:
        import diag_probe as probe
        control = [profile_control(a.port, 'start_profile', {'activities': ['CUDA_PROFILER'], 'num_steps': 100000, 'start_step': 0, 'output_dir': str(D.parent / 'worker-profile')})]
        probe.emit('capture_client_start')
        try:
            official.main(configs[1])
        finally:
            probe.emit('capture_client_stop')
            control.append(profile_control(a.port, 'stop_profile', {}))
            probe.flush()
            save(D / 'profile_control.json', control)
        check(control[-1]['start_unix'] - control[0]['end_unix'] <= preset['profile']['capture_seconds'], 'Capture duration exceeded')
        check(counter[0] == preset['warmup'] + preset['measured'], 'Diagnostic request count differs')
    else:
        official.main(configs[1])
    if finish_arrivals:
        finish_arrivals()
    check(len(traffic) == 2, 'Expected warmup and measured official calls')
    for phase, row in zip(['warmup', 'measured'], traffic):
        validate_result(read(D / (phase + '.json')), row['sampled_requests'])
    save(D / 'client_receipt.json', {'official_module': official.__file__, 'after_import_unix': imported, 'official_calls': 2, 'traffic': traffic, 'warmup_to_measurement_preparation_gap_s': traffic[1]['start_unix'] - traffic[0]['end_unix'], 'same_process_post_import_warmup': True})

if __name__ == '__main__':
    main()
