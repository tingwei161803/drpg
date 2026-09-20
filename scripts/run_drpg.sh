#!/usr/bin/env bash
# Run one method across all six benchmarks, one stream at a time.
#
#   scripts/run_drpg.sh                              # DRPG on gemini-2.0-flash
#   scripts/run_drpg.sh self_stream_icl llama-3.3-70b
#   WANDB_ENTITY=my-team scripts/run_drpg.sh         # also log to Weights & Biases
#
# Text-to-SQL needs the databases first: uv run download_text2sql_data.py
set -euo pipefail

METHOD="${1:-drpg}"
MODEL="${2:-gemini-2.0-flash}"
BENCHMARKS=(spider cosql bird hotpotqa ddxplus ds_1000)

AGENT_CFG="configs/agent/${METHOD}/${MODEL}.yml"
if [[ ! -f "$AGENT_CFG" ]]; then
    echo "no such config: $AGENT_CFG" >&2
    echo "available methods: $(ls configs/agent | tr '\n' ' ')" >&2
    exit 1
fi

WANDB_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS=(--use_wandb --entity "$WANDB_ENTITY")
fi

for bench in "${BENCHMARKS[@]}"; do
    echo "=== ${METHOD} / ${MODEL} / ${bench} ==="
    uv run python -m stream_bench.pipelines.run_bench \
        --agent_cfg "$AGENT_CFG" \
        --bench_cfg "configs/bench/${bench}.yml" \
        "${WANDB_ARGS[@]}"
done
