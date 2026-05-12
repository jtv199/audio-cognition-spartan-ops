#!/bin/bash
# Verify the alm-phi4mm env has everything it needs.
set +e

source /etc/profile.d/modules.sh
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1 Mambaforge/23.1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(conda info --base)/etc/profile.d/mamba.sh" 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim2341/kaip1/envs
mamba activate alm-phi4mm

echo "=== torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())' || echo "NOT INSTALLED"
echo; echo "=== transformers ==="
python -c 'import transformers; print("transformers", transformers.__version__)' || echo "NOT INSTALLED"
echo; echo "=== peft ==="
python -c 'import peft; print("peft", peft.__version__)' || echo "NOT INSTALLED"
echo; echo "=== flash_attn ==="
python -c 'import flash_attn; print("flash_attn", flash_attn.__version__)' || echo "NOT INSTALLED (falls back to eager)"
echo; echo "=== soundfile / scipy ==="
python -c 'import soundfile, scipy; print("soundfile", soundfile.__version__, "scipy", scipy.__version__)' || echo "NOT INSTALLED"

echo; echo "=== Phi-4-MM weights ==="
SNAP_DIR=/data/gpfs/projects/punim2341/kaip1/hf_cache/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots
if [ -d "$SNAP_DIR" ]; then
    SNAP=$(ls "$SNAP_DIR" | head -1)
    echo "snapshot: $SNAP_DIR/$SNAP"
    du -sh "$SNAP_DIR/$SNAP" 2>/dev/null
    ls "$SNAP_DIR/$SNAP" | head -10
else
    echo "weights NOT downloaded"
fi
