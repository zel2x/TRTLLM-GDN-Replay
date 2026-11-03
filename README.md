# GDN Prefill Replay

Reduce CPU launch overhead in TensorRT-LLM's GDN prefill path by capturing its
numerical operations once and replaying them as a CUDA graph.

The implementation keeps the existing arithmetic and uses the original eager
path when input layout or state ownership does not match the prepared graph.
Capture happens during initialization. The tested fast paths handle one pure
prefill request of 128 or 1024 tokens across the model's 18 GDN layers.

TL;DR:
 - TRT-LLM source code changes are in [patches](patches/)
 - Benchmark data and analysis in [RESULTS.md](RESULTS.md)
 - Other files are setup code and test helpers.

## Setup

Use a Linux CUDA environment matching `upstream.lock.json`, an SM120 GPU,
and a local copy of **Qwen3.5-0.8B**. The lock file identifies the model and
runtime. Set `MODEL` to the cached model directory.

```bash
export MODEL=/path/to/Qwen3.5-0.8B
python tools/fetch_upstream.py
python tools/prepare_candidate.py
python tools/verify.py --model "$MODEL"
```

Source is fetched into `.cache/upstream`. The patch is applied to isolated Python
modules in `.cache/prepared`; compiled operations come from the installed runtime.
The launcher selects those modules in both the server and its GPU workers.
Existing installations remain unchanged.

## Run a comparison

Preview the process and request counts:

```bash
python benchmarks/run.py c1
```

Run the comparison on an otherwise idle GPU:

```bash
python benchmarks/run.py c1 --model "$MODEL" --output artifacts/c1 --execute
python benchmarks/analyze.py --run artifacts/c1
```

Each process completes eight warmup requests before measurement. Stock and fix
are tested in alternating order. Both use the same FlashInfer launch-cache
control. Sampling is greedy, streaming is enabled, and EOS is ignored to preserve
output length. The official serving client supplies HTTP execution and metrics.
Every request is checked for success and actual token counts. 

| Preset | Input / output tokens | Traffic | Measured requests per process |
| --- | --- | --- | ---: |
| `c1` | 128 / 128 | Concurrency 1 | 200 |
| `c8-short` | 128 / 128 | Concurrency 8 | 50 |
| `c8-long` | 128 / 128 | Concurrency 8 | 200 |
| `c8-cohort` | 128 / 128 | Synchronized groups of 8 | 200 |
| `c1-1024` | 1024 / 128 | Concurrency 1 | 200 |
| `c8-1024` | 1024 / 128 | Concurrency 8 | 200 |

Replace `c1` in the command with any preset. Each preset selects the replay
implementation for its capture sizes. Use a fresh output directory for each
comparison.

The `c8-cohort` preset releases eight ready request tasks together, then waits
for all eight to finish before releasing the next group. It validates a maximum
5 ms spread between client request starts and records actual executor shapes.
Simultaneous submissions do not force a single eight-request prefill batch.
 Graphs are created only during initialization.

## Read the findings

[RESULTS.md](RESULTS.md) explains the measured gains, tail-latency variation and
replay coverage. Recalculate its performance tables from the bundled raw output:

```bash
python benchmarks/analyze.py --reference
```

TTFT is time to first token. TPOT is each request's remaining latency divided by
its remaining output tokens. ITL is the gap between received tokens; it is
aggregated over tokens. Output throughput counts generated tokens only.

Repeated stock/fix comparisons report paired 95% confidence intervals. The original short C1
reference uses log ratios; the other comparisons use arithmetic percentage
changes. The analysis preserves those original methods. Individual
comparisons report the observed change. Timing varies between runs; compare all
measurements and retain both improvements and regressions.

## Capture Nsight reports

```bash
python benchmarks/capture_nsys.py c8-long --model "$MODEL" \
  --nsys /path/to/nsys --output artifacts/c8-trace --execute
```

This captures stock and fix in separate runs. Worker-side controls start capture
after warmup.
CUDA, NVTX, OS-runtime events and CUDA-graph nodes are recorded. Each arm produces
`capture.nsys-rep`, a SQLite export, request markers and coverage checks.

Use unprofiled comparisons for performance numbers. Use the traces to explain
CPU dispatch, GPU work, graph replay and scheduling. Profiling can change queueing
and batch membership.

## Serve directly

```bash
python tools/run.py --variant fix --model "$MODEL" \
  --output artifacts/server --execute
```

The server binds to loopback. Choose `--variant stock` for the comparison arm.
The default fix uses the 128-token replay implementation. For the 1024-token
path, add `--snapshot prefill1024`; this also selects the matching `[128, 1024]`
outer-graph configuration.

## Project contents

- `patches/`: replay implementation and 1024-token extension.
- `benchmarks/`: frozen workloads, runtime configuration and runnable comparisons.
- `results/reference/`: de-identified measurements and numerical evidence.
- `upstream.lock.json`: machine-readable source, dependency and input identities.

Reference records replace account-specific model/tokenizer paths with the public model ID and use module-relative component paths. Timing samples, token counts and generated responses are unchanged. The reference index checksums identify these de-identified records.

Fetched source, prepared modules and generated reports are excluded from shared copies. Native profiler captures can contain host/network information; keep them private unless separately sanitized.
