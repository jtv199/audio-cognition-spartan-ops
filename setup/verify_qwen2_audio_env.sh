#!/bin/bash
# Verify the alm-qwen2-audio env has everything it needs.
set +e

source /etc/profile.d/modules.sh
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1 Mambaforge/23.1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(conda info --base)/etc/profile.d/mamba.sh" 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim2341/kaip1/envs
mamba activate alm-qwen2-audio

echo "=== torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())' || echo "NOT INSTALLED"
echo; echo "=== transformers ==="
python -c 'import transformers; print("transformers", transformers.__version__)' || echo "NOT INSTALLED"
echo; echo "=== Qwen2Audio classes ==="
python -c 'from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor; print("OK")' || echo "NOT IMPORTABLE"
echo; echo "=== flash_attn ==="
python -c 'import flash_attn; print("flash_attn", flash_attn.__version__)' || echo "NOT INSTALLED (perf only)"
echo; echo "=== soundfile ==="
python -c 'import soundfile; print("soundfile", soundfile.__version__)' || echo "NOT INSTALLED"

echo; echo "=== Qwen2-Audio weights ==="
SNAP_DIR=/data/gpfs/projects/punim2341/kaip1/hf_cache/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots
if [ -d "$SNAP_DIR" ]; then
    SNAP=$(ls "$SNAP_DIR" | head -1)
    echo "snapshot: $SNAP_DIR/$SNAP"
    du -sh "$SNAP_DIR/$SNAP" 2>/dev/null
    ls "$SNAP_DIR/$SNAP" | head -10
else
    echo "weights NOT downloaded"
fi
