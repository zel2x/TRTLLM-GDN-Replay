# SPDX-License-Identifier: Apache-2.0
"""Recompute request metrics and the original paired comparisons using only stdlib."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from common import check, read, save

T_DF4 = 2.7764451051977987
METRICS = ['mean_ttft_ms', 'p50_ttft_ms', 'p95_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'p50_tpot_ms', 'p95_tpot_ms', 'p99_tpot_ms', 'mean_e2el_ms', 'p50_e2el_ms', 'p95_e2el_ms', 'p99_e2el_ms', 'mean_itl_ms', 'p95_itl_ms', 'p99_itl_ms', 'output_throughput']

def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

def request_metrics(raw):
    count = raw['completed']
    check(count == len(raw['ttfts']) == len(raw['e2els']) == len(raw['output_lens']), 'Incomplete request measurements')
    check(not any(raw['errors']), 'Failed requests are present')
    tpots = [(latency - first) / (tokens - 1) for latency, first, tokens in zip(raw['e2els'], raw['ttfts'], raw['output_lens']) if tokens > 1]
    samples = {'ttft': raw['ttfts'], 'tpot': tpots, 'e2el': raw['e2els'], 'itl': [x for row in raw['itls'] for x in row]}
    result = {}
    for name, values in samples.items():
        check(bool(values), 'Missing metric samples: ' + name)
        result['mean_' + name + '_ms'] = statistics.mean(values) * 1000
        for p in [50, 95, 99]:
            result[f'p{p}_{name}_ms'] = percentile(values, p) * 1000
    result['output_throughput'] = sum(raw['output_lens']) / raw['duration']
    result['request_throughput'] = count / raw['duration']
    result['total_token_throughput'] = (sum(raw['input_lens']) + sum(raw['output_lens'])) / raw['duration']
    return result

def compare(pairs, method):
    check(len(pairs) in [1, 5], 'Comparison requires one or five process pairs')
    effects = {}
    for metric in METRICS:
        ratios = [pair['candidate'][metric] / pair['baseline'][metric] for pair in pairs]
        percentages = [100 * (x - 1) for x in ratios]
        values = [math.log(x) for x in ratios] if method == 'log_ratio' else percentages
        center = statistics.mean(values)
        convert = (lambda x: 100 * math.expm1(x)) if method == 'log_ratio' else (lambda x: x)
        row = {'change_percent': convert(center), 'per_pair_change_percent': percentages}
        if len(values) == 5:
            half = T_DF4 * statistics.stdev(values) / math.sqrt(5)
            row['paired_95_ci'] = [convert(center - half), convert(center + half)]
        effects[metric] = row
    return {'pairs': len(pairs), 'effect_method': method, 'effects': effects}

def load_raw(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == '.gz' else read(path)

def analyze_reference():
    index = read(ROOT / 'results/reference/index.json')
    result = {}
    for campaign in index['campaigns']:
        paired = {}
        observations = []
        for run in campaign['runs']:
            path = ROOT / run['file']
            entry = index['files'][run['file']]
            check(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'Reference archive hash mismatch')
            original = gzip.decompress(path.read_bytes())
            check(len(original) == entry['original_bytes'] and hashlib.sha256(original).hexdigest() == entry['original_sha256'], 'Original benchmark output hash mismatch')
            metrics = request_metrics(json.loads(original))
            paired.setdefault(run['pair'], {})[run['variant']] = metrics
            observations.append({'pair': run['pair'], 'variant': run['variant'], 'metrics': metrics})
        pairs = [paired[i] for i in sorted(paired) if set(paired[i]) == {'baseline', 'candidate'}]
        result[campaign['name']] = {'observations': observations}
        if pairs:
            result[campaign['name']].update(compare(pairs, campaign['effect_method']))
    return result

def analyze_run(directory):
    directory = Path(directory)
    manifest = read(directory / 'run.json')
    paired = {}
    for run in manifest['runs']:
        check(run['state'] == 'complete', 'Incomplete run: ' + run['label'])
        data = request_metrics(read(directory / run['label'] / 'client/measured.json'))
        paired.setdefault(run['pair'], {})[run['variant']] = data
    pairs = [paired[k] for k in sorted(paired)]
    check(all(set(x) == {'baseline', 'candidate'} for x in pairs), 'An arm is missing')
    result = compare(pairs, manifest['preset']['effect_method'])
    result['profiled'] = manifest['profiled']
    return result

def render(result):
    campaigns = result if 'effects' not in result else {'comparison': result}
    lines = ['| Benchmark | p95 TTFT change | Mean TPOT change | Output throughput change |', '|---|---:|---:|---:|']
    for name, row in campaigns.items():
        if 'effects' not in row:
            continue
        e = row['effects']
        lines.append(f"| {name} | {e['p95_ttft_ms']['change_percent']:+.2f}% | {e['mean_tpot_ms']['change_percent']:+.2f}% | {e['output_throughput']['change_percent']:+.2f}% |")
    return '\n'.join(lines) + '\n'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', action='store_true')
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    check(a.reference != bool(a.run), 'Choose --reference or --run')
    result = analyze_reference() if a.reference else analyze_run(a.run)
    if a.output:
        save(a.output, result)
    print(render(result), end='')

if __name__ == '__main__':
    main()
