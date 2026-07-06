#!/usr/bin/env bash
# NVIDIA Nemotron-3-Super-120B-A12B (NVFP4) on eugr/spark-vllm (GB10/SM121)
# Working config: all-Marlin NVFP4 path (Spark-safe), max 256K context, MTP-3,
# fastsafetensors load, super_v3 reasoning parser. No auto-restart (test).
# Requires Qwen down (memory). Model: hybrid Mamba-2 + Attention + MoE + MTP.
# See nemotron-report.md for the full story.
set -euo pipefail

MODEL=/models/nemotron3-super-120b-nvfp4
IMAGE=eugr/spark-vllm:latest
NAME=nemotron-test

docker rm -f "$NAME" 2>/dev/null || true

# entrypoint is /opt/nvidia/nvidia_entrypoint.sh → command starts with `vllm serve`
exec docker run -d --name "$NAME" --gpus all --net=host --ipc=host \
  -v /home/factory/models:/models \
  --restart no \
  --log-opt max-size=50m --log-opt max-file=5 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --served-model-name nvidia/nemotron-3-super nemotron \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --reasoning-parser-plugin /models/nemotron3-super-120b-nvfp4/super_v3_reasoning_parser.py \
  --reasoning-parser super_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --quantization fp4 \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --async-scheduling \
  --kernel-config '{"moe_backend": "marlin", "linear_backend": "marlin"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95}'
# NVIDIA's Spark env vars (VLLM_NVFP4_GEMM_BACKEND / VLLM_USE_FLASHINFER_MOE_FP4)
# were renamed in vLLM dev764 (rejected as unknown). The dev764-correct way to
# force the Spark-safe Marlin NVFP4 path is --kernel-config moe/linear=marlin.
# NVIDIA validates temp=1.0 + top_p=0.95 across ALL tasks for this (official) quant
# — unlike the community Leanstral NVFP4 which broke at 1.0.
