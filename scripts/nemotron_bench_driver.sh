#!/usr/bin/env bash
# Nemotron-3-Super-120B-A12B (NVFP4) MTP sweep on GB10, llama-benchy, same
# methodology as the Qwen study. Varies MTP num_speculative_tokens (3/1/0);
# prefix-caching pass on the MTP-3 baseline. Model/quant/kernels kept constant.
set -uo pipefail
IMAGE=eugr/spark-vllm:latest
MODEL=/models/nemotron3-super-120b-nvfp4
NAME=nemotron-test
OUT=/home/factory/nemotron-bench-results
BENCHY=/home/factory/benchy-venv/bin/llama-benchy
BENCHY_HF=/home/factory/benchy-hf
DEPTHS="0 512 1024 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536"
mkdir -p "$OUT"; rm -f "$OUT"/*.json "$OUT"/*.stdout.txt "$OUT"/*.deployfail.txt 2>/dev/null

deploy(){ # $1 nspec (0 = no speculation)
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  local SPEC=()
  [ "$1" != "0" ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$1,\"moe_backend\":\"triton\"}")
  docker run -d --name "$NAME" --gpus all --net=host --ipc=host \
    -v /home/factory/models:/models --restart no --log-opt max-size=50m --log-opt max-file=5 \
    -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE" vllm serve "$MODEL" \
    --served-model-name nvidia/nemotron-3-super nemotron --port 8000 --tensor-parallel-size 1 \
    --trust-remote-code --load-format fastsafetensors \
    --reasoning-parser-plugin /models/nemotron3-super-120b-nvfp4/super_v3_reasoning_parser.py --reasoning-parser super_v3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --quantization fp4 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 \
    --max-model-len 262144 --max-num-seqs 4 --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling \
    --kernel-config '{"moe_backend": "marlin", "linear_backend": "marlin"}' \
    "${SPEC[@]}" \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' >/dev/null
}
wait_healthy(){ for i in $(seq 1 90); do
  curl -sf -m3 http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
  [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" = exited ] && return 1
  sleep 10; done; return 1; }
bench(){ local label="$1"; shift; HF_HOME="$BENCHY_HF" "$BENCHY" --base-url http://127.0.0.1:8000/v1 --model nemotron \
  --depth $DEPTHS --pp 512 --tg 256 --exact-tg --runs 2 --latency-mode generation "$@" \
  --format json --save-result "$OUT/$label.json" > "$OUT/$label.stdout.txt" 2>&1; }

# label | nspec | prefix_pass
CONFIGS=("N_mtp3|3|1" "N_mtp1|1|0" "N_nospec|0|0")
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label nspec ppass <<< "$cfg"
  echo "=== $(date +%H:%M:%S) DEPLOY $label (mtp n=$nspec) ==="
  deploy "$nspec"
  if wait_healthy; then
    echo "$(date +%H:%M:%S) healthy -> bench $label"; bench "$label"
    [ "$ppass" = "1" ] && { echo "$(date +%H:%M:%S) prefix bench $label"; bench "${label}.pc" --enable-prefix-caching; }
    echo "$(date +%H:%M:%S) done $label"
  else echo "$(date +%H:%M:%S) DEPLOY FAILED $label"; docker logs --tail 40 "$NAME" > "$OUT/$label.deployfail.txt" 2>&1 || true; fi
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
