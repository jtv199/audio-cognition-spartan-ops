#!/bin/bash
# One-time bootstrap of the Qwen2-Audio-7B-Instruct mamba env + weights.
#
# RUN INSIDE: sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
#             tmux new -s setup
# (~30-45 min for the HF download)
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

ENV_NAME="alm-qwen2-audio"

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
    # Qwen2-Audio: Qwen2AudioForConditionalGeneration was added in transformers
    # 4.45.x. Pin <5.0 for the same reason Kimi-Audio needs it (transformers 5.x
    # is strict about missing config attrs in custom modeling code).
    pip install --no-cache-dir \
        "transformers>=4.45,<5.0" accelerate \
        soundfile librosa numpy \
        huggingface_hub

    echo "[setup] installing flash-attn (10-30 min build)"
    pip install --no-cache-dir packaging wheel ninja setuptools
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS=4 \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
        pip install --no-cache-dir --no-build-isolation flash-attn
fi

MODEL_REPO="Qwen/Qwen2-Audio-7B-Instruct"
SNAP_DIR="$HF_HUB_CACHE/models--${MODEL_REPO//\//--}/snapshots"

WEIGHT_COUNT=0
if [ -d "$SNAP_DIR" ]; then
    WEIGHT_COUNT=$(find "$SNAP_DIR" -maxdepth 2 -name "*.safetensors" 2>/dev/null | wc -l)
fi
if [ "$WEIGHT_COUNT" -ge 1 ]; then
    echo "[setup] $MODEL_REPO weights present ($WEIGHT_COUNT safetensors) — skipping"
else
    echo "[setup] downloading $MODEL_REPO (~17 GB)"
    hf download "$MODEL_REPO"
fi

echo "[setup] smoke test"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor; print('Qwen2Audio classes OK')"
python -c "import soundfile; print('soundfile', soundfile.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash_attn NOT installed (perf only — model will still run)"

echo
echo "DONE. Smoke-test the inference path with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  MANIFEST=\$PWD/pc-benchmark-pilot/manifests/smoke_1sample_manifest.jsonl \\"
echo "  OUT=\$PWD/runs/smoke_qwen2_audio.raw.jsonl \\"
echo "  sbatch external/spartan-ops/sbatch/run_qwen2_audio.sbatch"
