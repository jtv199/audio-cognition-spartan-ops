#!/bin/bash
# One-time bootstrap of the Step-Audio-2-mini mamba env + weights.
#
# Step-Audio-2-mini (stepfun-ai/Step-Audio-2-mini) is an 8B end-to-end
# multi-modal LLM from StepFun. Ga 51.1 on MMAU. Apache 2.0 licence.
#
# Differs from the other 5 ALMs: it does NOT use a standard transformers
# from_pretrained() path. Instead the official inference example wraps
# StepAudio2 from `stepaudio2` (in their GitHub repo, cloned alongside).
# So setup must (a) install deps, (b) clone Step-Audio2 repo for the
# `stepaudio2` Python package, (c) download the HF weights snapshot.
#
# RUN INSIDE: sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
#             tmux new -s setup
# (~45-60 min)
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

ENV_NAME="alm-step-audio2"
STEP_REPO="$PROJ/Step-Audio2"

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

    echo "[setup] installing pinned deps (per stepfun-ai/Step-Audio-2-mini HF card)"
    # transformers==4.49.0 is STRICT per the model card. Step-Audio-2's custom
    # modeling code targets this exact version. s3tokenizer + diffusers +
    # hyperpyyaml are required by the audio codec pipeline. onnxruntime is
    # used by the audio tokenizer.
    pip install --no-cache-dir \
        transformers==4.49.0 \
        torchaudio librosa \
        onnxruntime \
        s3tokenizer \
        diffusers \
        hyperpyyaml \
        soundfile numpy scipy \
        huggingface_hub sentencepiece protobuf accelerate

    # flash-attn is NOT in the official requirements list, but Step-Audio-2
    # would benefit from it. Skip the build for now — too risky to gate
    # smoke-test on a 20-min compile. Can add later if perf becomes an issue.
fi

# ─── Clone StepFun's Step-Audio2 repo for the stepaudio2 Python package ─
if [ -d "$STEP_REPO/.git" ]; then
    echo "[setup] $STEP_REPO already cloned — pulling latest"
    (cd "$STEP_REPO" && git pull --ff-only 2>&1 | tail -3 || true)
else
    echo "[setup] cloning stepfun-ai/Step-Audio2 into $STEP_REPO"
    git clone --depth 1 https://github.com/stepfun-ai/Step-Audio2.git "$STEP_REPO"
fi

# Add the cloned repo to the env's site-packages so `import stepaudio2` works.
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$STEP_REPO" > "$SITE_PACKAGES/step_audio2.pth"
echo "[setup] wrote .pth pointer at $SITE_PACKAGES/step_audio2.pth"

# ─── Pre-stage Step-Audio-2-mini weights ─────────────────────────────
MODEL_REPO="stepfun-ai/Step-Audio-2-mini"
SNAP_DIR="$HF_HUB_CACHE/models--${MODEL_REPO//\//--}/snapshots"

WEIGHT_COUNT=0
if [ -d "$SNAP_DIR" ]; then
    WEIGHT_COUNT=$(find "$SNAP_DIR" -maxdepth 2 -name "*.safetensors" 2>/dev/null | wc -l)
fi
if [ "$WEIGHT_COUNT" -ge 1 ]; then
    echo "[setup] $MODEL_REPO weights present ($WEIGHT_COUNT safetensors) — skipping"
else
    echo "[setup] downloading $MODEL_REPO (~16 GB)"
    hf download "$MODEL_REPO"
fi

# StepAudio2('Step-Audio-2-mini') expects a model directory NAME relative to
# the working directory, NOT a HF cache path — it does internal config
# resolution. Symlink the HF snapshot to a friendlier path so the runner
# can pass `Step-Audio-2-mini` literally as the model identifier.
LATEST_SNAP=$(ls -d "$SNAP_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_SNAP" ]; then
    SYMLINK_TARGET="$STEP_REPO/Step-Audio-2-mini"
    if [ ! -e "$SYMLINK_TARGET" ]; then
        ln -s "${LATEST_SNAP%/}" "$SYMLINK_TARGET"
        echo "[setup] symlinked $SYMLINK_TARGET -> ${LATEST_SNAP%/}"
    else
        echo "[setup] $SYMLINK_TARGET already exists — leaving"
    fi
fi

echo "[setup] smoke test"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torchaudio, librosa, soundfile; print('audio libs OK')"
python -c "import s3tokenizer, diffusers, hyperpyyaml, onnxruntime; print('step-audio deps OK')"
(cd "$STEP_REPO" && python -c "import stepaudio2; print('stepaudio2 OK at', stepaudio2.__file__)") || echo "stepaudio2 NOT IMPORTABLE — check Step-Audio2 repo layout"

echo
echo "DONE. Smoke-test the inference path with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  MANIFEST=\$PWD/pc-benchmark-pilot/manifests/smoke_1sample_manifest.jsonl \\"
echo "  OUT=\$PWD/runs/smoke_step_audio2.raw.jsonl \\"
echo "  sbatch external/spartan-ops/sbatch/run_step_audio2_mini.sbatch"
