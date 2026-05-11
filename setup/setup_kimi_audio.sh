#!/bin/bash
# One-time bootstrap of the Kimi-Audio mamba env + weights.
#
# RUN INSIDE: sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
#             tmux new -s setup
# (~30-60 min for the HF download, depending on network)
#
# Idempotent: re-running checks for existing env + weights and skips if found.

set -euo pipefail

# ─── Refuse to run on a login node ────────────────────────────────────
case "$(hostname)" in
    *login*)
        echo "ERROR: setup_kimi_audio.sh must NOT run on a login node ($(hostname))."
        echo "       Get on a compute node first:"
        echo "         sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G"
        exit 1
        ;;
esac

# ─── Shared env + modules ────────────────────────────────────────────
# Slurm's `sbatch <file>` copies the script to /var/spool/slurm/jobN/slurm_script
# which breaks BASH_SOURCE[0]-relative resolution. Handle 3 cases in order:
#   1. SPARTAN_OPS_DIR env-var override (explicit)
#   2. Real script location via BASH_SOURCE if it resolves to a file with env.sh nearby
#   3. Conventional project path under punim2341 (fallback for the slurm-copy case)
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
if [ ! -f "$REPO_DIR/env.sh" ]; then
    echo "ERROR: cannot find env.sh under REPO_DIR=$REPO_DIR" >&2
    echo "       Set SPARTAN_OPS_DIR=/path/to/external/spartan-ops and re-run." >&2
    exit 1
fi
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$HF_HOME" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" \
         "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$PROJ/slurm_logs" "$PROJ/runs"

ENV_NAME="alm-kimi"

# ─── Create / reuse the mamba env ────────────────────────────────────
if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] mamba env $ENV_NAME already exists — activating"
    mamba activate "$ENV_NAME"
else
    echo "[setup] creating mamba env $ENV_NAME (python 3.10)"
    mamba create -y -n "$ENV_NAME" python=3.10 pip
    mamba activate "$ENV_NAME"

    echo "[setup] installing PyTorch with CUDA 12.1 wheels"
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

    echo "[setup] installing core deps"
    pip install --no-cache-dir \
        "transformers>=4.45" accelerate \
        soundfile librosa numpy \
        huggingface_hub

    echo "[setup] installing kimia_infer from MoonshotAI/Kimi-Audio repo"
    if ! python -c "import kimia_infer" 2>/dev/null; then
        pip install --no-cache-dir \
            "git+https://github.com/MoonshotAI/Kimi-Audio.git#egg=kimia_infer" \
        || {
            echo "[setup] git+pip failed — falling back to clone + editable install"
            cd "$PROJ"
            rm -rf "$PROJ/Kimi-Audio"
            git clone --depth 1 https://github.com/MoonshotAI/Kimi-Audio.git "$PROJ/Kimi-Audio"
            cd "$PROJ/Kimi-Audio"
            pip install --no-cache-dir -e .
        }
    fi
fi

# ─── Pre-stage model weights ─────────────────────────────────────────
MODEL_REPO="moonshotai/Kimi-Audio-7B-Instruct"
SNAP_DIR="$HF_HUB_CACHE/models--${MODEL_REPO//\//--}/snapshots"

if [ -d "$SNAP_DIR" ] && [ -n "$(ls -A "$SNAP_DIR" 2>/dev/null)" ]; then
    echo "[setup] $MODEL_REPO weights already present in $HF_HUB_CACHE — skipping download"
else
    echo "[setup] pre-downloading $MODEL_REPO weights to $HF_HOME (~30-60 min)"
    hf download "$MODEL_REPO" \
        --include "*.safetensors" "*.json" "tokenizer*" "*.txt" "*.py"
fi

# ─── Smoke test (does kimia_infer import?) ───────────────────────────
echo "[setup] smoke test"
python -c "import kimia_infer; print('kimia_infer OK at', kimia_infer.__file__)"
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

echo
echo "DONE. Submit the inference job with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  MANIFEST=\$PWD/pc-benchmark-pilot/manifests/timit_single_phoneme_15.manifest.jsonl \\"
echo "  OUT=\$PWD/runs/kimi_timit_single_phoneme_15_raw.jsonl \\"
echo "  sbatch external/spartan-ops/sbatch/run_kimi_audio.sbatch"
