#!/bin/bash
# Run Kimi-Audio-7B-Instruct on the ACPT manifest (Keith 1994).
#
# 20 trials × ~3.3 min single-syllable word stream; subject counts 'dog' targets.
# MAX_NEW_TOKENS=32 — answer is a single integer.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_acpt_kimi.sbatch

#SBATCH -J acpt_kimi
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

PROJ_BASE="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
MANIFEST="${1:-${MANIFEST:-$PROJ_BASE/pc-benchmark-pilot/manifests/acpt_manifest.jsonl}}"
OUT="${OUT:-$PROJ_BASE/runs/kimi_acpt.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MODE="${MODE:-direct}"
REASON_MAX_WORDS="${REASON_MAX_WORDS:-20}"

if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    REPO_DIR="$PROJ_BASE/external/spartan-ops"
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
    sys.exit("ERROR: no Kimi-Audio snapshot — run setup/setup_kimi_audio.sh")
print(snaps[-1])
PY
)

echo "[$(date -Is)] ACPT Kimi-Audio"
echo "  job_id:          ${SLURM_JOB_ID:-(none)}"
echo "  node:            $(hostname)"
echo "  manifest:        $MANIFEST"
echo "  output:          $OUT"
echo "  model:           $MODEL_REF"
echo "  max_items:       $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

python "$JACK_BENCH/kimi_official_benchmark.py" \
    --model-dir "$MODEL_REF" \
    --manifest "$MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --log-every 2 \
    --mode "$MODE" \
    --reason-max-words "$REASON_MAX_WORDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --retry-max-new-tokens 32 \
    --empty-retry 2 \
    --merge-gap-sec 0.25 \
    --output "$OUT"

echo "[$(date -Is)] done -> $OUT"
