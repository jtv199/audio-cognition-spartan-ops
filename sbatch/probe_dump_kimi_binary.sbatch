#!/bin/bash
# Binary single-interval BCGD encoder dump — Kimi-Audio-7B.
# Uses probe_dump_encoder_binary.py (one wav per item, saves full (T×D) hidden state).
# Companion to probe_dump_encoder_kimi.sbatch (3AFC version).
#
# SUBMIT:
#   PROJ=/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark
#   MANIFEST=$PROJ/pc-benchmark-pilot/manifests/bcgd_probe_binary_manifest.jsonl \
#   OUT_DIR=$PROJ/runs/probe_embeddings_binary \
#   sbatch external/spartan-ops/sbatch/probe_dump_kimi_binary.sbatch

#SBATCH -J probe_kimi_binary
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/bcgd_probe_binary_manifest.jsonl}"
OUT_DIR="${OUT_DIR:?set OUT_DIR=/path/to/runs/probe_embeddings_binary}"

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
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" "$OUT_DIR"

mamba activate alm-kimi

PROBE_SCRIPT="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/tools/probe_dump_encoder_binary.py"
SELF_TEST="${SELF_TEST:-0}"

# Resolve the Kimi snapshot directory.
export MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--moonshotai--Kimi-Audio-7B-Instruct", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    print(f"ERROR: no Kimi snapshot under {pat}", file=sys.stderr); sys.exit(1)
print(snaps[-1])
PY
)

# kimia_infer pydeps (ships alongside the model snapshot).
KIMIA_PYDEPS_GUESS="$(dirname "$MODEL_REF")/pydeps"
if [ -d "$KIMIA_PYDEPS_GUESS" ]; then
    export PYTHONPATH="$KIMIA_PYDEPS_GUESS:${PYTHONPATH:-}"
fi

echo "[$(date -Is)] starting kimi BINARY encoder dump"
echo "  manifest:    $MANIFEST"
echo "  out_dir:     $OUT_DIR/kimi"
echo "  MODEL_REF:   $MODEL_REF"
echo "  PYTHONPATH:  ${PYTHONPATH:-(unset)}"
echo "  self_test:   $SELF_TEST"
nvidia-smi -L 2>&1 | head -4

EXTRA=()
if [ "$SELF_TEST" -gt 0 ]; then EXTRA+=(--self-test "$SELF_TEST"); fi

python "$PROBE_SCRIPT" \
    --model kimi \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR" \
    "${EXTRA[@]}"

echo "[$(date -Is)] finished -> $OUT_DIR/kimi"
