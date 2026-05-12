#!/bin/bash
# Verify the alm-step-audio2 env has everything it needs.
set +e

source /etc/profile.d/modules.sh
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1 Mambaforge/23.1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(conda info --base)/etc/profile.d/mamba.sh" 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim2341/kaip1/envs
mamba activate alm-step-audio2

echo "=== torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())' || echo "NOT INSTALLED"
echo; echo "=== transformers (must be 4.49.0) ==="
python -c 'import transformers; print("transformers", transformers.__version__)' || echo "NOT INSTALLED"
echo; echo "=== step-audio deps ==="
python -c 'import s3tokenizer, diffusers, hyperpyyaml, onnxruntime; print("OK")' || echo "NOT INSTALLED"
echo; echo "=== audio libs ==="
python -c 'import torchaudio, librosa, soundfile; print("OK")' || echo "NOT INSTALLED"
echo; echo "=== stepaudio2 module ==="
cd /data/gpfs/projects/punim2341/kaip1/Step-Audio2 2>/dev/null
python -c 'import stepaudio2; print("OK at", stepaudio2.__file__)' || echo "NOT IMPORTABLE"

echo; echo "=== Step-Audio-2-mini weights ==="
SNAP_DIR=/data/gpfs/projects/punim2341/kaip1/hf_cache/hub/models--stepfun-ai--Step-Audio-2-mini/snapshots
if [ -d "$SNAP_DIR" ]; then
    SNAP=$(ls "$SNAP_DIR" | head -1)
    echo "snapshot: $SNAP_DIR/$SNAP"
    du -sh "$SNAP_DIR/$SNAP" 2>/dev/null
    ls "$SNAP_DIR/$SNAP" | head -10
else
    echo "weights NOT downloaded"
fi

echo; echo "=== Step-Audio-2-mini symlink ==="
ls -la /data/gpfs/projects/punim2341/kaip1/Step-Audio2/Step-Audio-2-mini 2>/dev/null || echo "symlink NOT present"
