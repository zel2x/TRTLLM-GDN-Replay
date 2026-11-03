# Findings

## Performance

All rows below are unprofiled HTTP-serving measurements using Qwen3.5-0.8B in BF16.
Changes compare replay against stock with the same common FlashInfer control.
Negative latency changes mean faster; positive throughput changes mean faster.

| Experiment | p95 TTFT change | Mean TPOT change | Output throughput change |
| --- | ---: | ---: | ---: |
| C1 | **−35.26%** | +0.22% | +2.09% |
| C8, 50 requests | −29.36% | +1.37% | +1.28% |
| C8, 200 requests | +14.24% | +1.99% | −1.37% |
| C8, synchronized arrivals | −7.88% | −0.38% | +1.96% |
| C1, 1024/128 | -5.53% | -1.42% | +1.53% |
| C8, 1024/128 | -0.34% | -1.06% | +1.10% |

The 128-token C1 p95 TTFT reduction has a paired 95% interval of **32.09%–38.28%**.

Recalculate every row and the available confidence intervals:

```bash
python benchmarks/analyze.py --reference
```

### Median latency (p50)

p50 is the median request latency. For repeated comparisons, the displayed
times are averages of per-run medians. Percentage changes use the original
paired calculation: log ratios for short C1 and arithmetic percentage changes
for the other setups.

| Setup | p50 TTFT stock → fix | TTFT change | p50 TPOT change | p50 completion-latency change |
| --- | ---: | ---: | ---: | ---: |
| C1, 128/128 | 48.46 → 31.34 ms | −35.33% | +0.02% | −2.19% |
| C8, 50 requests | 113.22 → 86.36 ms | −23.72% | +1.56% | −2.06% |
| C8, 200 requests | 105.03 → 95.55 ms | −8.77% | +1.86% | +1.18% |
| C8, synchronized arrivals | 113.69 → 94.57 ms | −16.71% | −0.46% | −2.13% |
| C1, 1024/128 | 60.25 → 58.98 ms | −2.11% | −0.37% | −0.65% |
| C8, 1024/128 | 307.11 → 305.50 ms | −0.52% | −1.42% | −0.89% |

## Why 128-token C1 benefits most

At concurrency one, each request reaches an eligible pure 128-token prefill.
Replay removes repeated Python dispatch across 18 GDN layers. It does not change
the numerical kernels. TTFT falls because the first token waits for prefill.
The remaining 127 tokens use the existing decode path, so TPOT stays nearly flat
and whole-request latency improves by about 2%.

At higher concurrency, prefills often share an iteration with decoding. Those
mixed iterations follow the original path. Faster initial prefill can also move
the next request-fetch boundary, changing which requests enter a batch.

## Unsynchronized C8 arrivals

We observed regression in this run, +14.24% increase in ttft. The root cause is
scheduled batch shape changes. With GDN subgraph replay, the first batch's
prefill time decreases, this may result as the next batch to be scheduled before
all 8 requests arrive. In the actual run, if one batch arrived late we see
there is one more mixed prefill/decode batch and an extra single request
decode batch. This regression is gone if we synchronize the request arrival in
the same batch, details in below.

## Synchronized C8 arrivals

The client released groups of eight ready requests, waited for the entire group
to finish, and then released the next group. Requests retained 128 input and 128
output tokens. The largest validated client-start spread was **4.42 ms**.

Mean TTFT fell **13.32%**; p95 TTFT fell **7.88%**, with a paired 95% change
interval of **−15.34% to −0.41%**.

GPU kernel counts were identical and kernel busy time changed by
about 0.12%. First-prefill GDN host time fell from **19.21 to 2.97 ms per group**.


## 1,024-token inputs

Each request contains 1,024 input tokens and generates 128 tokens. Each process
uses eight warmup and 200 measured requests. C1 keeps one request outstanding;
C8 releases eight requests together and waits for all eight before the next group.

Both stock and candidate capture outer prefill graphs for `[128, 1024]`. The
candidate additionally captures the unchanged GDN numerical region for each size.
The 1,024-token forward budget, sampling, arithmetic and KV capacity stay equal.

| Workload | Measured 1,024-token GDN layer replays | Prefills using the child graph |
| --- | ---: | ---: |
| C1, 1024/128 | 3,600 | 200/200 |
| C8, 1024/128 | 450 | 25/200 |

Each eligible prefill replays all 18 GDN layers. Mixed prefill/decode iterations
use the original GDN execution. C8 therefore benefits less directly from the new
path, even though requests are released together.

Reproduce the comparisons:

```bash
python benchmarks/run.py c1-1024 --model "$MODEL" --output artifacts/long-c1 --execute
python benchmarks/run.py c8-1024 --model "$MODEL" --output artifacts/long-c8 --execute
```

## Correctness

The component checks cover state reuse, changed inputs and slots, stream handling,
unsupported-input fallback and cache-manager replacement. Matched-history model
checks recorded zero maximum logit normalized-RMS error, identical mean next-token
loss and 128/128 raw top-1 agreements. The reference numerical summary is stored
alongside the performance measurements.
Long-prompt evidence: [numerical summary](results/reference/numerical-1024-summary.json).

## Interpretation

This is a targeted reduction in GDN prefill host dispatch. Its benefit depends on
how often eligible pure-prefill execution occurs. Read TTFT, decode latency,
throughput, tail latency and replay coverage together rather than treating one
favorable number as a general inference speedup.

Note: A similar effort has been merged recently by PR #20815
