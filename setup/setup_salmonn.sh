#!/bin/bash
# One-time bootstrap of the SALMONN-13B mamba env + weights + bytedance/SALMONN repo.
#
# SALMONN-13B (tsinghua-ee/SALMONN):
#   LoRA-on-Vicuna-13B-v1.1 with a window-level Q-Former that fuses
#   Whisper-large-v2 (speech) + BEATs_iter3+ (audio events) encoders.
#   Audio-only input (no image/video). Custom code — NOT transformers-native.
#
# Required artefacts (downloaded to $HF_HUB_CACHE / $PROJ/models):
#   1. bytedance/SALMONN repo on branch `salmonn` (cli_inference.py, models/, configs/)
#   2. openai/whisper-large-v2                          (~3 GB)
#   3. WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2  (BEATs cpt2, ~360 MB)
#      (HF mirror of the OneDrive cpt2 ckpt the official repo points at)
#   4. lmsys/vicuna-13b-v1.1                            (~26 GB)
#   5. tsinghua-ee/SALMONN/salmonn_v1.pth               (~400 MB LoRA + Q-Former ckpt)
#
# RUN INSIDE: sbatch external/spartan-ops/sbatch/setup_salmonn.sbatch
# (Or interactive: sinteractive -p interactive --time=06:00:00 -c 4 --mem=24G ; tmux)
# (~30-60 min env build + ~30 GB download = 2-4 hr wall on Spartan)
#
# Idempotent.

set -euo pipefail

case "$(hostname)" in
    *login*)
        echo "ERROR: do not run on a login node ($(hostname))."
        echo "       sbatch external/spartan-ops/sbatch/setup_salmonn.sbatch"
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
         "$PROJ/slurm_logs" "$PROJ/runs" "$PROJ/models"

ENV_NAME="alm-salmonn"

# ─── Clone bytedance/SALMONN (salmonn branch — SALMONN-13B inference code) ─
SALMONN_SRC="$PROJ/models/SALMONN-src"
if [ -d "$SALMONN_SRC/.git" ]; then
    echo "[setup] SALMONN repo already cloned at $SALMONN_SRC — pulling"
    git -C "$SALMONN_SRC" fetch --depth=1 origin salmonn || true
    git -C "$SALMONN_SRC" checkout salmonn || true
    git -C "$SALMONN_SRC" pull --ff-only || true
else
    echo "[setup] cloning bytedance/SALMONN @ salmonn -> $SALMONN_SRC"
    git clone --depth=1 --branch salmonn https://github.com/bytedance/SALMONN.git "$SALMONN_SRC"
fi
[ -f "$SALMONN_SRC/cli_inference.py" ] || { echo "ERROR: SALMONN clone missing cli_inference.py" >&2; exit 1; }

# ─── Mamba env ────────────────────────────────────────────────────────────
if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] mamba env $ENV_NAME already exists"
    mamba activate "$ENV_NAME"
else
    echo "[setup] creating mamba env $ENV_NAME (python 3.9 — SALMONN requirement)"
    mamba create -y -n "$ENV_NAME" python=3.9 pip
    mamba activate "$ENV_NAME"

    # SALMONN's requirements.txt pins torch==2.0.1 / transformers==4.28.0, but:
    #   - torch 2.0.1+cu118 → bitsandbytes 0.41.3 → triton.ops ModuleNotFoundError
    #     (triton dropped triton.ops; 0.41.x bnb still imports it).
    #   - torch 2.0.1+cu118 + bitsandbytes 0.39.1 → CUDA 12.4 driver vs cu118
    #     bnb lib "Setup Failed" mismatch on Spartan A100s.
    # Upgrading to torch 2.1.2 cu121 wheel sidesteps both: cu121 driver-compatible
    # under forward compat with CUDA 12.4, and bitsandbytes 0.42.0 ships
    # pre-built cu121 binaries and works on torch 2.1.x without triton.ops.
    # transformers stays at 4.28.0 — SALMONN's models/ code reads internals from
    # that exact minor (e.g. LlamaModel forward signature).
    echo "[setup] installing PyTorch 2.1.2 (cu121 wheel — CUDA 12.4 driver fwd-compatible)"
    pip install --no-cache-dir \
        torch==2.1.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

    echo "[setup] installing SALMONN requirements (pinned)"
    pip install --no-cache-dir \
        peft==0.3.0 \
        transformers==4.28.0 \
        sentencepiece==0.1.97 \
        accelerate==0.20.3 \
        bitsandbytes==0.42.0 \
        soundfile librosa numpy scipy \
        huggingface_hub omegaconf einops timm pyyaml protobuf
fi

# ─── Pre-stage Whisper-large-v2 (~3 GB) ──────────────────────────────────
WHISPER_REPO="openai/whisper-large-v2"
WHISPER_SNAP_DIR="$HF_HUB_CACHE/models--${WHISPER_REPO//\//--}/snapshots"
if [ -d "$WHISPER_SNAP_DIR" ] && [ -n "$(find "$WHISPER_SNAP_DIR" -maxdepth 3 -name 'preprocessor_config.json' 2>/dev/null)" ]; then
    echo "[setup] $WHISPER_REPO present — skipping"
else
    echo "[setup] downloading $WHISPER_REPO (~3 GB)"
    hf download "$WHISPER_REPO"
fi

# ─── Pre-stage Vicuna-13B-v1.1 (~26 GB) ───────────────────────────────────
VICUNA_REPO="lmsys/vicuna-13b-v1.1"
VICUNA_SNAP_DIR="$HF_HUB_CACHE/models--${VICUNA_REPO//\//--}/snapshots"
VICUNA_SHARDS=0
if [ -d "$VICUNA_SNAP_DIR" ]; then
    VICUNA_SHARDS=$(find "$VICUNA_SNAP_DIR" -maxdepth 3 \( -name '*.bin' -o -name '*.safetensors' \) 2>/dev/null | wc -l)
fi
if [ "$VICUNA_SHARDS" -ge 1 ]; then
    echo "[setup] $VICUNA_REPO weights present ($VICUNA_SHARDS shards) — skipping"
else
    echo "[setup] downloading $VICUNA_REPO (~26 GB — 20-40 min on Spartan)"
    hf download "$VICUNA_REPO"
fi

# ─── Pre-stage SALMONN-13B LoRA + Q-Former ckpt (~400 MB) ─────────────────
SALMONN_REPO="tsinghua-ee/SALMONN"
SALMONN_SNAP_DIR="$HF_HUB_CACHE/models--${SALMONN_REPO//\//--}/snapshots"
if [ -d "$SALMONN_SNAP_DIR" ] && [ -n "$(find "$SALMONN_SNAP_DIR" -maxdepth 3 -name 'salmonn_v1.pth' 2>/dev/null)" ]; then
    echo "[setup] $SALMONN_REPO present — skipping"
else
    echo "[setup] downloading $SALMONN_REPO (~400 MB)"
    hf download "$SALMONN_REPO"
fi

# ─── Pre-stage BEATs cpt2 (~360 MB) — HF mirror of official OneDrive ──────
BEATS_REPO="WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
BEATS_SNAP_DIR="$HF_HUB_CACHE/models--${BEATS_REPO//\//--}/snapshots"
if [ -d "$BEATS_SNAP_DIR" ] && [ -n "$(find "$BEATS_SNAP_DIR" -maxdepth 3 -name '*.pt' 2>/dev/null)" ]; then
    echo "[setup] $BEATS_REPO present — skipping"
else
    echo "[setup] downloading $BEATS_REPO (~360 MB)"
    hf download "$BEATS_REPO"
fi

# ─── Resolve absolute paths to artefacts and emit a salmonn_paths.env ──
WHISPER_PATH=$(find "$WHISPER_SNAP_DIR" -maxdepth 3 -name 'preprocessor_config.json' | head -1 | xargs -I{} dirname {})
VICUNA_PATH=$(find "$VICUNA_SNAP_DIR" -maxdepth 3 -name 'config.json' | head -1 | xargs -I{} dirname {})
SALMONN_CKPT=$(find "$SALMONN_SNAP_DIR" -maxdepth 3 -name 'salmonn_v1.pth' | head -1)
BEATS_CKPT=$(find "$BEATS_SNAP_DIR" -maxdepth 3 -name '*.pt' | head -1)

cat > "$PROJ/models/salmonn_paths.env" <<EOF
# Auto-generated by setup_salmonn.sh — sourced by run_*_salmonn.sbatch.
export SALMONN_SRC="$SALMONN_SRC"
export SALMONN_WHISPER_PATH="$WHISPER_PATH"
export SALMONN_VICUNA_PATH="$VICUNA_PATH"
export SALMONN_CKPT_PATH="$SALMONN_CKPT"
export SALMONN_BEATS_PATH="$BEATS_CKPT"
EOF
echo "[setup] wrote $PROJ/models/salmonn_paths.env"
cat "$PROJ/models/salmonn_paths.env"

echo "[setup] smoke test (imports only — full model load deferred to first GPU job)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import peft; print('peft', peft.__version__)"
python -c "from transformers import WhisperFeatureExtractor; print('WhisperFeatureExtractor OK')"
python -c "import soundfile, librosa; print('soundfile', soundfile.__version__, 'librosa', librosa.__version__)"

# Sanity: SALMONN repo importable when on sys.path
( cd "$SALMONN_SRC" && python -c "import sys; sys.path.insert(0,'.'); from models.salmonn import SALMONN; print('SALMONN class import OK')" ) || \
    echo "[setup] WARN: SALMONN class import failed — inspect $SALMONN_SRC/models/"

echo
echo "DONE. Smoke-test inference with:"
echo "  cd $REPO_DIR/.. && \\"
echo "  sbatch external/spartan-ops/sbatch/run_ravlt_salmonn.sbatch"
