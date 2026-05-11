# Shared environment exports for audio-cognition-spartan-ops.
# Source from setup/*.sh and sbatch/*.sbatch with: `source "$REPO/env.sh"`.
#
# Sets:
#   PROJ            project work root (under /data/gpfs/projects/punim2341/kaip1)
#   REPO            audio-cognition-benchmark clone under $PROJ
#   JACK_BENCH      external/jack_benchmark submodule (runner scripts)
#   HF_HOME         HuggingFace cache root, redirected off $HOME
#   CONDA_ENVS_PATH where mamba envs live
#   XDG_CACHE_HOME  per-job ephemeral cache, redirected off $HOME
#   HOME            redirected so conda/torch/pip never write to real /home/kaip1
#   TMPDIR          per-job tmp, redirected
#
# After sourcing, mkdir -p $HF_HOME $XDG_CACHE_HOME $HOME $TMPDIR before use.

PROJ=/data/gpfs/projects/punim2341/kaip1
REPO="$PROJ/audio-cognition-benchmark"
JACK_BENCH="$REPO/external/jack_benchmark"

export HF_HOME="$PROJ/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export CONDA_ENVS_PATH="$PROJ/envs"
export CONDA_PKGS_DIRS="$PROJ/.conda_pkgs"

# Use SLURM_JOB_ID if inside a job (so concurrent jobs don't collide);
# otherwise use $$ for the interactive shell.
_JOB_TAG="${SLURM_JOB_ID:-${USER}_$$}"
export XDG_CACHE_HOME="$PROJ/.cache/$_JOB_TAG"
export HF_MODULES_CACHE="$XDG_CACHE_HOME/hf_modules"
export TORCH_HOME="$PROJ/.cache/torch"
export HOME="$PROJ/.home"
export TMPDIR="$PROJ/.tmp/$_JOB_TAG"
