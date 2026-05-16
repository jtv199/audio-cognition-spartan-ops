#!/bin/bash
# Run Phi-4-multimodal-instruct on the quicksin manifest.
#
# Submit (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_quicksin_phi4mm.sbatch
# Optional env overrides: MANIFEST, OUT, MAX_NEW_TOKENS, MAX_ITEMS, DISABLE_CONSTRAINED.

#SBATCH -J quicksin_phi4mm
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
MANIFEST="${MANIFEST:-${1:-$PROJ_BASE/pc-benchmark-pilot/manifests/quicksin_manifest.jsonl}}"
OUT="${OUT:-$PROJ_BASE/runs/phi4mm_quicksin.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
# Echo-loop mitigation: keep constrained decoding ENABLED by default
# (set DISABLE_CONSTRAINED=1 to override).
DISABLE_CONSTRAINED="${DISABLE_CONSTRAINED:-0}"

if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    REPO_DIR="$PROJ_BASE/external/spartan-ops"
fi
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR"          "$(dirname "$OUT")"

mamba activate alm-phi4mm

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--microsoft--Phi-4-multimodal-instruct", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    sys.exit("ERROR: no Phi-4-multimodal-instruct snapshot — run setup/setup_phi4mm.sh")
print(snaps[-1])
PY
)

echo "[$(date -Is)] Phi-4-MM quicksin"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  model:     $MODEL_REF"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS  disable_constrained: $DISABLE_CONSTRAINED"
nvidia-smi -L 2>&1 | head -4

EXTRA_ARGS=()
if [ "$DISABLE_CONSTRAINED" = "1" ]; then
    EXTRA_ARGS+=(--disable-constrained-decoding)
fi

python "$JACK_BENCH/phi4mm_official_benchmark.py"     --model-dir "$MODEL_REF"     --manifest "$MANIFEST"     --output "$OUT"     --max-items "$MAX_ITEMS"     --max-new-tokens "$MAX_NEW_TOKENS"     --prompt-fallback 'Describe this audio briefly.'     --log-every 10     "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] done -> $OUT"
