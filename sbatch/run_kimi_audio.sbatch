#!/bin/bash
# Run Kimi-Audio-7B-Instruct against a manifest on Spartan.
#
# SUBMIT:
#   MANIFEST=/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/pc-benchmark-pilot/manifests/timit_single_phoneme_15.manifest.jsonl \
#   OUT=/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/runs/kimi_timit_single_phoneme_15_raw.jsonl \
#   sbatch external/spartan-ops/sbatch/run_kimi_audio.sbatch
#
# Prerequisites: setup/setup_kimi_audio.sh has been run successfully
# (env exists at $PROJ/envs/alm-kimi, weights cached in $PROJ/hf_cache).

#SBATCH -J kimi_audio_timit
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err
# To enable email notifications, uncomment both lines and set address:
# #SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=jack.peng@student.unimelb.edu.au

set -euo pipefail

# ─── Inputs (overrideable via env-var) ───────────────────────────────
MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.jsonl}"
OUT="${OUT:?set OUT=/path/to/output.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"            # 0 = all
MODE="${MODE:-reasoned}"
REASON_MAX_WORDS="${REASON_MAX_WORDS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

# ─── Shared env + modules ────────────────────────────────────────────
# `sbatch <file>` copies the script into /var/spool/slurm/jobN/slurm_script,
# so BASH_SOURCE[0] no longer points at the submodule. Resolve REPO_DIR
# robustly: env-var override → BASH_SOURCE if env.sh is nearby → conventional
# project path fallback.
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
if [ ! -f "$REPO_DIR/env.sh" ]; then
    echo "ERROR: cannot find env.sh under REPO_DIR=$REPO_DIR" >&2
    exit 1
fi
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$(dirname "$OUT")"

ENV_NAME="alm-kimi"
mamba activate "$ENV_NAME"

# ─── Resolve model snapshot ──────────────────────────────────────────
MODEL_REPO="moonshotai/Kimi-Audio-7B-Instruct"
MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pattern = os.path.join(hub, "models--moonshotai--Kimi-Audio-7B-Instruct", "snapshots", "*")
snaps = sorted(glob.glob(pattern))
if not snaps:
    print(f"ERROR: no Kimi-Audio snapshot under {pattern}", file=sys.stderr)
    print("Run setup/setup_kimi_audio.sh first.", file=sys.stderr)
    sys.exit(1)
print(snaps[-1])
PY
)

echo "[$(date -Is)] starting Kimi-Audio inference"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  model:     $MODEL_REF"
echo "  max_items: $MAX_ITEMS  mode: $MODE  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

python "$JACK_BENCH/kimi_official_benchmark.py" \
    --model-dir "$MODEL_REF" \
    --manifest "$MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --log-every 20 \
    --mode "$MODE" \
    --reason-max-words "$REASON_MAX_WORDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --retry-max-new-tokens 32 \
    --empty-retry 2 \
    --merge-gap-sec 0.25 \
    --output "$OUT"

echo "[$(date -Is)] finished — wrote $OUT"

# ─── Sample-id passthrough check ─────────────────────────────────────
# The replay provider on the laptop side keys on sample_id. The
# kimi_official_benchmark.py runner copies all manifest fields onto each
# output row, so sample_id should be present. Quick sanity check:
if ! python -c "
import json, sys
with open('$OUT') as f:
    first = json.loads(f.readline())
sid = first.get('sample_id')
if not sid:
    print('WARNING: first output row has no sample_id', file=sys.stderr)
    sys.exit(2)
print(f'first sample_id: {sid}')
"; then
    echo "WARNING: predictions JSONL missing sample_id. The replay provider will not find these."
    echo "         tools/predictions_to_replay.py expects sample_id as the lookup key."
fi
