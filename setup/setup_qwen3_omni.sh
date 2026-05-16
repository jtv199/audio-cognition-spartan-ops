#!/bin/bash
# One-time bootstrap of the Qwen3-Omni-30B-A3B-Instruct mamba env + weights.
#
# Qwen3-Omni-30B-A3B-Instruct (Qwen/Qwen3-Omni-30B-A3B-Instruct):
#   30B-parameter MoE Omni model, A3B (3B active). Native multimodal (audio,
#   image, video, text). We use thinker-text-only mode (talker disabled) to
#   match the rest of the audio-cognition battery.
#
# Single HF repo:
#   1. Qwen/Qwen3-Omni-30B-A3B-Instruct (~60 GB safetensors)
#
# RUN INSIDE: sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
#             tmux new -s setup
# (~60-90 min — env build + 60 GB download)
#
# Idempotent.

set -euo pipefail

case "$(hostname)" in
    *login*)
        echo "ERROR: do not run on a login node ($(hostname))."
        echo "       sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G"
        exit 1 ;;
esac

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

mkdir -p "$HF_HOME" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" \
         "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$PROJ/slurm_logs" "$PROJ/runs"

ENV_NAME="alm-qwen3-omni"

if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] mamba env $ENV_NAME already exists"
    mamba activate "$ENV_NAME"
else
    echo "[setup] creating mamba env $ENV_NAME (python 3.10)"
    mamba create -y -n "$ENV_NAME" python=3.10 pip
    mamba activate "$ENV_NAME"

    echo "[setup] installing PyTorch (cu121)"
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

    # Qwen3OmniMoeForConditionalGeneration + Qwen3OmniMoeProcessor were merged
    # to transformers main in Sep-Oct 2025; first stable release with full
    # support is 4.57.x. Pin >=4.57 to ensure the classes resolve.
    echo "[setup] installing transformers (>=4.57 for Qwen3-Omni)"
    pip install --no-cache-dir \
        "transformers>=4.57,<5.0" accelerate \
        soundfile librosa numpy scipy \
        huggingface_hub sentencepiece protobuf

    # qwen_omni_utils provides process_mm_info() — same API the Qwen2.5-Omni /
    # Omni-R1 runners use. >=0.0.4 returns the 4-tuple shape.
    echo "[setup] installing qwen-omni-utils"
    pip install --no-cache-dir 'qwen-omni-utils>=0.0.4'

    echo "[setup] installing flash-attn (10-30 min build)"
    pip install --no-cache-dir packaging wheel ninja setuptools
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS=4 \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
        pip install --no-cache-dir --no-build-isolation flash-attn || \
        echo "[setup] WARN: flash-attn build failed; runner will fall back to sdpa"
fi

# ─── Pre-stage Qwen3-Omni-30B-A3B-Instruct weights (~60 GB) ──────────────
MODEL_REPO="Qwen/Qwen3-Omni-30B-A3B-Instruct"
SNAP_DIR="$HF_HUB_CACHE/models--${MODEL_REPO//\//--}/snapshots"

WEIGHT_COUNT=0
if [ -d "$SNAP_DIR" ]; then
    WEIGHT_COUNT=$(find "$SNAP_DIR" -maxdepth 2 -name "*.safetensors" 2>/dev/null | wc -l)
fi
if [ "$WEIGHT_COUNT" -ge 1 ]; then
    echo "[setup] $MODEL_REPO weights present ($WEIGHT_COUNT safetensors) — skipping"
else
    echo "[setup] downloading $MODEL_REPO (~60 GB — 30-60 min on Spartan)"
    hf download "$MODEL_REPO"
fi

echo "[setup] smoke test"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor; print('Qwen3-Omni classes OK')"
python -c "from qwen_omni_utils import process_mm_info; print('qwen_omni_utils OK')"
python -c "import soundfile; print('soundfile', soundfile.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash_attn NOT installed (runner will use sdpa)"

echo
echo "DONE. Smoke-test the inference path with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  sbatch external/spartan-ops/sbatch/run_ravlt_qwen3_omni.sbatch"
