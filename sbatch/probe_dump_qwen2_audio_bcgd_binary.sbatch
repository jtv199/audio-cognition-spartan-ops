#!/bin/bash
# Binary single-interval BCGD encoder dump — Qwen2-Audio.
#
# SUBMIT (example):
#   MANIFEST=/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/pc-benchmark-pilot/manifests/bcgd_probe_binary_manifest.jsonl \
#   OUT_DIR=/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/runs/probe_embeddings_binary \
#   sbatch external/spartan-ops/sbatch/probe_dump_qwen2_audio_bcgd_binary.sbatch

#SBATCH -J probe_qwen2_audio_bcgd_binary
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

mamba activate alm-qwen2-audio

PROJ="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
PROBE_SCRIPT="$PROJ/tools/probe_dump_encoder_binary.py"

echo "[$(date -Is)] starting qwen2_audio binary BCGD encoder dump"
echo "  manifest: $MANIFEST"
echo "  out_dir:  $OUT_DIR/qwen2_audio"
nvidia-smi -L 2>&1 | head -4

# Encoder: model.audio_tower (Whisper-large-v3, D=1280, 50 Hz frame rate).
# Binary single-interval: one wav per item, label = 0 (no-gap) or 1 (gap).
python "$PROBE_SCRIPT" \
    --model qwen2_audio \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR"

echo "[$(date -Is)] finished -> $OUT_DIR/qwen2_audio"
