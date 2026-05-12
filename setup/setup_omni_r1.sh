#!/bin/bash
# One-time bootstrap of the Omni-R1 mamba env + weights.
#
# Omni-R1 (Haoz0206/Omni-R1) is an RL-tuned Qwen2.5-Omni-Thinker checkpoint
# from the "Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?"
# paper (arXiv 2505.09439). Top MMAU Ga in that paper at 54.5.
#
# Two HF repos are needed:
#   1. Haoz0206/Omni-R1 — the fine-tuned thinker weights (used as --model-dir)
#   2. Qwen/Qwen2.5-Omni-7B — the base model that provides Qwen2_5OmniProcessor
#      (the runner loads model from #1 but processor from #2)
#
# RUN INSIDE: sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
#             tmux new -s setup
# (~45-75 min — two model downloads)
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

ENV_NAME="alm-omni-r1"

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

    echo "[setup] installing core deps"
    # Qwen2_5OmniThinkerForConditionalGeneration + Qwen2_5OmniProcessor were
    # added to transformers in 4.50.x. Pin >=4.50 to be safe; keep <5.0 to
    # avoid the rope_theta-style breakage Kimi hit.
    pip install --no-cache-dir \
        "transformers>=4.50,<5.0" accelerate \
        soundfile librosa numpy scipy \
        huggingface_hub sentencepiece protobuf

    # qwen_omni_utils ships the process_mm_info() helper that omni_r1_local_smoke.py
    # imports. Available on PyPI as `qwen-omni-utils`.
    echo "[setup] installing qwen-omni-utils"
    pip install --no-cache-dir qwen-omni-utils

    echo "[setup] installing flash-attn (10-30 min build)"
    pip install --no-cache-dir packaging wheel ninja setuptools
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS=4 \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
        pip install --no-cache-dir --no-build-isolation flash-attn
fi

# ─── Pre-stage Omni-R1 fine-tuned weights ─────────────────────────────
MODEL_REPO="Haoz0206/Omni-R1"
SNAP_DIR="$HF_HUB_CACHE/models--${MODEL_REPO//\//--}/snapshots"

WEIGHT_COUNT=0
if [ -d "$SNAP_DIR" ]; then
    WEIGHT_COUNT=$(find "$SNAP_DIR" -maxdepth 2 -name "*.safetensors" 2>/dev/null | wc -l)
fi
if [ "$WEIGHT_COUNT" -ge 1 ]; then
    echo "[setup] $MODEL_REPO weights present ($WEIGHT_COUNT safetensors) — skipping"
else
    echo "[setup] downloading $MODEL_REPO (~15 GB)"
    hf download "$MODEL_REPO"
fi

# ─── Pre-stage Qwen2.5-Omni-7B base (for processor + chat_template) ────
BASE_REPO="Qwen/Qwen2.5-Omni-7B"
BASE_SNAP_DIR="$HF_HUB_CACHE/models--${BASE_REPO//\//--}/snapshots"

BASE_HAS_PROCESSOR=0
if [ -d "$BASE_SNAP_DIR" ]; then
    BASE_HAS_PROCESSOR=$(find "$BASE_SNAP_DIR" -maxdepth 2 -name "preprocessor_config.json" 2>/dev/null | wc -l)
fi
if [ "$BASE_HAS_PROCESSOR" -ge 1 ]; then
    echo "[setup] $BASE_REPO processor present — skipping"
else
    echo "[setup] downloading $BASE_REPO (~17 GB — only processor files needed, but"
    echo "        hf download takes the whole repo by default)"
    hf download "$BASE_REPO"
fi

echo "[setup] smoke test"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor; print('Qwen2.5-Omni classes OK')"
python -c "from qwen_omni_utils import process_mm_info; print('qwen_omni_utils OK')"
python -c "import soundfile; print('soundfile', soundfile.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash_attn NOT installed (runner needs flash_attention_2 for default attn impl)"

echo
echo "DONE. Smoke-test the inference path with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  MANIFEST=\$PWD/pc-benchmark-pilot/manifests/smoke_1sample_manifest.jsonl \\"
echo "  OUT=\$PWD/runs/smoke_omni_r1.raw.jsonl \\"
echo "  sbatch external/spartan-ops/sbatch/run_omni_r1.sbatch"
