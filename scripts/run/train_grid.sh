#!/usr/bin/env bash
# Trains one MA-Chunk configuration over a grid of token prices and seeds, one process per
# (lambda, seed), in parallel. Each process uses one CPU thread.
#
# Usage (from the repository root):
#   scripts/run/train_grid.sh <crag|hotpotqa|musique> <ma|ma_msg|single> "<lambdas>" "<seeds>" [extra train_ma flags]
# Examples:
#   scripts/run/train_grid.sh hotpotqa ma_msg "0.3 1.0" "0 1 2"
#   scripts/run/train_grid.sh crag ma_msg "0.4 1.25" "0 1 2" --reward-mode marginal_ap --turn-order confidence
# Environment: JOBS (parallel processes, default 8), OUT (default results/validation/<dataset>/runs),
#              TIMESTEPS (per fold, default 200000).
set -euo pipefail
DS=$1; VARIANT=$2; LAMS=$3; SEEDS=$4; shift 4
JOBS=${JOBS:-8}; TIMESTEPS=${TIMESTEPS:-200000}
OUT=${OUT:-results/validation/$DS/runs}
POOL=(); [ "$DS" != "crag" ] && POOL=(--pool "data/$DS/candidates.parquet")
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
for L in $LAMS; do for S in $SEEDS; do
  echo python -m ma_chunk.train_ma "${POOL[@]}" --variant "$VARIANT" --lams "$L" --seeds "$S" \
       --timesteps "$TIMESTEPS" --out-dir "$OUT" "$@"
done; done | xargs -P "$JOBS" -I{} sh -c "{}"
