#!/bin/bash
# One-time bootstrap of the Phi-4-Multimodal mamba env + weights.
#
# RUN INSIDE: sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
#             tmux new -s setup
# (~30-45 min)
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

ENV_NAME="alm-phi4mm"

if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] mamba env $ENV_NAME already exists"
    mamba activate "$ENV_NAME"
else
    echo "[setup] creating mamba env $ENV_NAME (python 3.10)"
    mamba create -y -n "$ENV_NAME" python=3.10 pip
    mamba activate "$ENV_NAME"

    # Phi-4-MM's HF page pins exact versions (discussion thread #58). Reproduce
    # them. torch==2.6.0 needs cu124 wheel index (cu121 wheel is 2.5.x max).
    echo "[setup] installing PyTorch 2.6.0 (cu124)"
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

    echo "[setup] installing pinned deps (per microsoft/Phi-4-multimodal-instruct)"
    pip install --no-cache-dir \
        transformers==4.48.2 \
        accelerate==1.3.0 \
        peft==0.13.2 \
        soundfile==0.13.1 \
        scipy==1.15.2 \
        pillow==11.1.0 \
        backoff==2.2.1 \
        numpy librosa \
        huggingface_hub sentencepiece protobuf

    echo "[setup] installing flash-attn 2.7.4.post1 (10-30 min build)"
    pip install --no-cache-dir packaging wheel ninja setuptools
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS=4 \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
        pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
fi

MODEL_REPO="microsoft/Phi-4-multimodal-instruct"
SNAP_DIR="$HF_HUB_CACHE/models--${MODEL_REPO//\//--}/snapshots"

WEIGHT_COUNT=0
if [ -d "$SNAP_DIR" ]; then
    WEIGHT_COUNT=$(find "$SNAP_DIR" -maxdepth 2 -name "*.safetensors" 2>/dev/null | wc -l)
fi
if [ "$WEIGHT_COUNT" -ge 1 ]; then
    echo "[setup] $MODEL_REPO weights present ($WEIGHT_COUNT safetensors) — skipping"
else
    echo "[setup] downloading $MODEL_REPO (~11 GB)"
    hf download "$MODEL_REPO"
fi

echo "[setup] smoke test"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import peft; print('peft', peft.__version__)"
python -c "import soundfile; print('soundfile', soundfile.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash_attn NOT installed — runner falls back to eager"

echo
echo "DONE. Smoke-test the inference path with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  MANIFEST=\$PWD/pc-benchmark-pilot/manifests/smoke_1sample_manifest.jsonl \\"
echo "  OUT=\$PWD/runs/smoke_phi4mm.raw.jsonl \\"
echo "  sbatch external/spartan-ops/sbatch/run_phi4mm.sbatch"
