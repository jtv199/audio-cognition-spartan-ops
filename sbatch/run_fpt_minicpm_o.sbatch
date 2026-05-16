#!/bin/bash
# Run MiniCPM-o-2_6 on the fpt manifest.
#
# Submit (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_fpt_minicpm_o.sbatch
# Optional env overrides: MANIFEST, OUT, MAX_NEW_TOKENS, MAX_ITEMS.

#SBATCH -J fpt_minicpm_o
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

PROJ_BASE="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
MANIFEST="${MANIFEST:-${1:-$PROJ_BASE/pc-benchmark-pilot/manifests/fpt_manifest.jsonl}}"
OUT="${OUT:-$PROJ_BASE/runs/minicpm_o_fpt.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    REPO_DIR="$PROJ_BASE/external/spartan-ops"
fi
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" "$(dirname "$OUT")"

mamba activate alm-minicpm-o

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--openbmb--MiniCPM-o-2_6", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    sys.exit("ERROR: no MiniCPM-o-2_6 snapshot — run setup/setup_minicpm_o.sh")
print(snaps[-1])
PY
)

echo "[$(date -Is)] MiniCPM-o FPT"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  model:     $MODEL_REF"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

python "$JACK_BENCH/minicpm_o_official_benchmark.py" \
    --manifest "$MANIFEST" \
    --model-dir "$MODEL_REF" \
    --output "$OUT" \
    --task-name "fpt" \
    --default-prompt 'Describe this audio briefly.' \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-items "$MAX_ITEMS"

echo "[$(date -Is)] done -> $OUT"
