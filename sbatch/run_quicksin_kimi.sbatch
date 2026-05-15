#!/bin/bash
# Run Kimi-Audio-7B-Instruct on the QuickSIN-style manifest.
#
# Mirrors run_hint_kimi.sbatch but tuned for QuickSIN-style:
#   - short sentences -> MAX_NEW_TOKENS=128
#   - output to runs/kimi_quicksin.raw.jsonl
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_quicksin_kimi.sbatch \
#       $PWD/pc-benchmark-pilot/manifests/quicksin_manifest.jsonl
#
# Optional env overrides: OUT, MAX_NEW_TOKENS, MAX_ITEMS, MODE.

#SBATCH -J kimi_quicksin
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

MANIFEST="${1:-${MANIFEST:?usage: sbatch run_quicksin_kimi.sbatch <manifest.jsonl>}}"
OUT="${OUT:-$PWD/runs/kimi_quicksin.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MODE="${MODE:-direct}"
REASON_MAX_WORDS="${REASON_MAX_WORDS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

# Resolve spartan-ops repo dir robustly (BASH_SOURCE[0] -> /var/spool under sbatch).
if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    _bs="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    if [ -n "$_bs" ] && [ -f "$_bs/../env.sh" ]; then
        REPO_DIR="$(cd "$_bs/.." && pwd)"
    else
        REPO_DIR="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/external/spartan-ops"
    fi
fi
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" "$(dirname "$OUT")"

mamba activate alm-kimi

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
snaps = sorted(glob.glob(os.path.join(hub, "models--moonshotai--Kimi-Audio-7B-Instruct", "snapshots", "*")))
if not snaps:
    print("ERROR: no Kimi-Audio snapshot", file=sys.stderr); sys.exit(1)
print(snaps[-1])
PY
)

echo "[$(date -Is)] Kimi QuickSIN-style: manifest=$MANIFEST out=$OUT max_new_tokens=$MAX_NEW_TOKENS"

python "$JACK_BENCH/kimi_official_benchmark.py" \
    --model-dir "$MODEL_REF" \
    --manifest "$MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --log-every 10 \
    --mode "$MODE" \
    --reason-max-words "$REASON_MAX_WORDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --retry-max-new-tokens 32 \
    --empty-retry 2 \
    --merge-gap-sec 0.25 \
    --output "$OUT"

echo "[$(date -Is)] done -> $OUT"

# sample_id passthrough sanity check
if ! python -c "
import json, sys
with open('$OUT') as f:
    first = json.loads(f.readline())
sid = first.get('sample_id')
if not sid:
    print('WARNING: first output row has no sample_id', file=sys.stderr); sys.exit(2)
print(f'first sample_id: {sid}')
"; then
    echo "WARNING: predictions JSONL missing sample_id."
fi
