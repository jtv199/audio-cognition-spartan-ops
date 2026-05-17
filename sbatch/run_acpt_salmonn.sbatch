#!/bin/bash
# Run SALMONN-13B on the ACPT manifest (Auditory Continuous Performance Test (Keith)).
#
# Submit (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_acpt_salmonn.sbatch
# Optional env overrides: MANIFEST, OUT, MAX_NEW_TOKENS, MAX_ITEMS.

#SBATCH -J acpt_salmonn
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
MANIFEST="${MANIFEST:-${1:-$PROJ_BASE/pc-benchmark-pilot/manifests/acpt_manifest.jsonl}}"
OUT="${OUT:-$PROJ_BASE/runs/salmonn_acpt.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    REPO_DIR="$PROJ_BASE/external/spartan-ops"
fi
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

# Load SALMONN artefact paths (written by setup_salmonn.sh).
PATHS_FILE="$PROJ/models/salmonn_paths.env"
[ -f "$PATHS_FILE" ] || { echo "ERROR: $PATHS_FILE missing - run setup_salmonn.sbatch" >&2; exit 1; }
source "$PATHS_FILE"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" "$(dirname "$OUT")"

mamba activate alm-salmonn

echo "[$(date -Is)] SALMONN-13B ACPT"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  whisper:   $SALMONN_WHISPER_PATH"
echo "  vicuna:    $SALMONN_VICUNA_PATH"
echo "  beats:     $SALMONN_BEATS_PATH"
echo "  ckpt:      $SALMONN_CKPT_PATH"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

python "$JACK_BENCH/salmonn_official_benchmark.py"     --manifest "$MANIFEST"     --output "$OUT"     --task-name "acpt"     --default-prompt 'Listen to the stream of words. Count how many times you heard the word '''dog'''. Answer with a single integer.'     --max-new-tokens "$MAX_NEW_TOKENS"     --max-items "$MAX_ITEMS"

echo "[$(date -Is)] done -> $OUT"
