#!/bin/bash
# Run Phi-4-Multimodal-Instruct against a manifest on Spartan.
#
# SUBMIT:
#   MANIFEST=$PWD/pc-benchmark-pilot/manifests/<name>.jsonl \
#   OUT=$PWD/runs/phi4mm_<name>.raw.jsonl \
#   sbatch external/spartan-ops/sbatch/run_phi4mm.sbatch
#
# Prerequisites: setup/setup_phi4mm.sh has been run successfully.

#SBATCH -J phi4mm
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

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.jsonl}"
OUT="${OUT:?set OUT=/path/to/output.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DISABLE_CONSTRAINED="${DISABLE_CONSTRAINED:-0}"

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

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$(dirname "$OUT")"

mamba activate alm-phi4mm

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--microsoft--Phi-4-multimodal-instruct", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    print(f"ERROR: no Phi-4-MM snapshot under {pat}", file=sys.stderr)
    print("Run setup/setup_phi4mm.sh first.", file=sys.stderr)
    sys.exit(1)
print(snaps[-1])
PY
)

echo "[$(date -Is)] starting Phi-4-MM inference"
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

python "$JACK_BENCH/phi4mm_official_benchmark.py" \
    --model-dir "$MODEL_REF" \
    --manifest "$MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --log-every 10 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$OUT" \
    "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] finished — wrote $OUT"
