# DGX Spark (GB10) — NVIDIA Nemotron-3-Super-120B-A12B (NVFP4) benchmarks

Single-stream inference benchmarks for **NVIDIA Nemotron-3-Super-120B-A12B** (hybrid
Mamba-2 + attention + MoE, NVFP4) served with **vLLM + MTP speculative decoding** on a single
**NVIDIA GB10 / DGX Spark**. Measured with the standard
**[llama-benchy](https://github.com/eugr/llama-benchy)** tool. Sister study to
[dgx-spark-qwen3.5-122b-bench](https://github.com/fank/dgx-spark-qwen3.5-122b-bench) — same
methodology, so the two models are directly comparable.

**Headline:** prefill/TTFT-bound at agent context sizes (TTFT ≈ 1.6 s @0 → 25 s @16k → 50 s @32k).
**MTP speculative decoding gives +49% decode** (18.6 vs 12.5 tok/s), with n=3 ≥ n=1. Prefix
caching works (3.6× TTFT @6k; higher depths unmeasured — see caveats). **Compared to
Qwen3.5-122B/DFlash on the same box, Nemotron is slower on both decode (~16 vs ~22 tok/s) and
TTFT (~50 s vs ~37 s @32k)** — its appeal is capability / 1M context, not speed. Full analysis in
[`comparison.md`](comparison.md).

## Environment

| | |
|---|---|
| **Model** | `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` — hybrid Mamba-2 + attention + MoE (`model_type: nemotron_h`), 88 layers, `max_position_embeddings` 262144 |
| **Quantization** | **NVFP4** (4-bit) + FP8, vLLM `modelopt` / `modelopt_mixed`; ~75 GB on disk |
| **Speculation** | **MTP** (built-in Multi-Token-Prediction head, no separate drafter), `num_speculative_tokens` swept 3/1/0, draft MoE backend `triton` |
| **Serving engine** | vLLM **`0.23.1rc1.dev764+g54b16d8a9`** |
| **Container image** | `eugr/spark-vllm:latest` (`sha256:d0840ff0e0ba1899a51bf4cb473f43d0c765288b8de708080ad9d95768615141`) |
| **Kernels** | all-Marlin NVFP4 path — `--kernel-config {"moe_backend":"marlin","linear_backend":"marlin"}`, `VLLM_TEST_FORCE_FP8_MARLIN=1` (Spark-safe; avoids the FlashInfer-Cutlass FP4 path) |
| **GPU** | NVIDIA **GB10** (DGX Spark class), SM **12.1**, 128 GB / 119 GiB unified LPDDR5X, ~273 GB/s |
| **Driver / CUDA** | 580.159.03 / CUDA 13.0 |
| **OS / kernel** | Ubuntu 24.04.4 LTS, kernel `6.17.0-1021-nvidia`, aarch64 (Grace) |
| **Benchmark tool** | `llama-benchy` 0.4.0 |
| **Serving params (fixed)** | `--kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 --max-model-len 262144 --max-num-seqs 4 --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling`, reasoning parser `super_v3`, tool-call parser `qwen3_coder` |
| **Benchmark params** | `--pp 512 --tg 256 --exact-tg --runs 2 --latency-mode generation`, depths `0 512 1024 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536` |
| **Date** | 2026-07-06 |

## Configs tested

| config | speculation | isolates |
|---|---|---|
| `N_mtp3` | MTP n=3 (baseline / shipped recipe) | reference |
| `N_mtp1` | MTP n=1 | MTP draft depth |
| `N_nospec` | none | what MTP buys |

Everything else constant (NVFP4, fp8 KV, Marlin kernels, 262k, util 0.85, prefix caching on).

## Contents

- [`comparison.md`](comparison.md) — verdict, per-config explanation, tables, graphs, prefix-cache analysis, Qwen comparison
- [`dataset.csv`](dataset.csv) / [`dataset.json`](dataset.json) — every datapoint (3 configs × 14 depths)
- `graphs/` — decode/TTFT/prefill vs context, grouped bars, cold-vs-cached TTFT
- `results/` — raw `llama-benchy` JSON (`N_mtp3.pc.json` = `--enable-prefix-caching` pass)
- `scripts/` — benchmark driver, aggregation/plotting, serve launcher

## Reproduce

```bash
python3 -m venv benchy-venv && ./benchy-venv/bin/pip install -U llama-benchy
bash scripts/nemotron_bench_driver.sh          # redeploys per MTP config, sweeps 14 depths
python3 scripts/aggregate_and_plot.py results . # -> CSV/JSON/markdown/graphs
```

## Caveats

Single-stream (concurrency=1), `runs=2` (some decode noise). The MTP-3 config OOM'd on its
first deploy (transient — memory still freeing from the previously-running model) and was
re-run cleanly. Prefix-cache cached-phase timing is **null above ~6k depth** here — an
async-scheduling / block-streaming interaction with llama-benchy, not a serving failure — so
high-depth cache speedup is unmeasured. Numbers are specific to this image/model/hardware/date.
