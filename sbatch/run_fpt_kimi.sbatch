#!/bin/bash
# Run Kimi-Audio-7B-Instruct on the FPT (Frequency Pattern Test) manifest.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_fpt_kimi.sbatch
#
# Optional env overrides: MANIFEST, OUT, MAX_NEW_TOKENS, MAX_ITEMS, MODE.

#SBATCH -J kimi_fpt
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

# ─── Absolute paths — not subject to env.sh PROJ clobber ─────────────────────
REPO_ABS="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
MANIFEST="${MANIFEST:-${REPO_ABS}/pc-benchmark-pilot/manifests/fpt_manifest.jsonl}"
OUT="${OUT:-${REPO_ABS}/runs/kimi_fpt.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MODE="${MODE:-direct}"
REASON_MAX_WORDS="${REASON_MAX_WORDS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

# ─── Shared env + modules ─────────────────────────────────────────────────────
if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    _bs="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    if [ -n "$_bs" ] && [ -f "$_bs/../env.sh" ]; then
        REPO_DIR="$(cd "$_bs/.." && pwd)"
    else
        REPO_DIR="${REPO_ABS}/external/spartan-ops"
    fi
fi
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$(dirname "$OUT")"

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

echo "[$(date -Is)] Kimi FPT: manifest=$MANIFEST out=$OUT max_new_tokens=$MAX_NEW_TOKENS"

python "$JACK_BENCH/kimi_official_benchmark.py" \
    --model-dir "$MODEL_REF" \
    --manifest "$MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --log-every 20 \
    --mode "$MODE" \
    --reason-max-words "$REASON_MAX_WORDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --retry-max-new-tokens 32 \
    --empty-retry 2 \
    --merge-gap-sec 0.25 \
    --output "$OUT"

echo "[$(date -Is)] done -> $OUT"
