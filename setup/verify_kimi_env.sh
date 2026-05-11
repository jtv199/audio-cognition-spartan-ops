#!/bin/bash
# Verify the alm-kimi env has everything it needs.
# Run from anywhere with `bash external/spartan-ops/setup/verify_kimi_env.sh`.

set +e

source /etc/profile.d/modules.sh
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1 Mambaforge/23.1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(conda info --base)/etc/profile.d/mamba.sh" 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim2341/kaip1/envs
mamba activate alm-kimi

echo "=== flash_attn ==="
python -c 'import flash_attn; print("flash_attn", flash_attn.__version__, "at", flash_attn.__file__)' || echo "NOT INSTALLED"

echo
echo "=== kimia_infer ==="
python -c 'import kimia_infer; print("kimia_infer at", kimia_infer.__file__)' || echo "NOT INSTALLED"

echo
echo "=== torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())' || echo "NOT INSTALLED"

echo
echo "=== transformers ==="
python -c 'import transformers; print("transformers", transformers.__version__)' || echo "NOT INSTALLED"

echo
echo "=== Kimi weights ==="
SNAP_DIR=/data/gpfs/projects/punim2341/kaip1/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots
if [ -d "$SNAP_DIR" ]; then
    SNAP=$(ls "$SNAP_DIR" | head -1)
    echo "snapshot: $SNAP_DIR/$SNAP"
    du -sh "$SNAP_DIR/$SNAP"
    ls "$SNAP_DIR/$SNAP" | head -10
else
    echo "weights NOT downloaded"
fi
