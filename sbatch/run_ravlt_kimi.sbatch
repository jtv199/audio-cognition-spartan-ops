#!/bin/bash
# Run Kimi-Audio-7B-Instruct on the RAVLT manifest (30 identical Trial-1 replicates).
#
# Rey (1958) RAVLT Trial-1 free recall: 15-word List A read aloud at 1 word/sec.
# TTS-synthesized stimuli (edge-tts, en-US-JennyNeural, 16 kHz).
# MAX_NEW_TOKENS=128 allows the model to list up to 15 words.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_ravlt_kimi.sbatch \
#       /data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark/pc-benchmark-pilot/manifests/ravlt_manifest.jsonl

#SBATCH -J ravlt_kimi
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

PROJ_BASE="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
MANIFEST="${1:-${MANIFEST:-$PROJ_BASE/pc-benchmark-pilot/manifests/ravlt_manifest.jsonl}}"
OUT="${OUT:-$PROJ_BASE/runs/kimi_ravlt.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MODE="${MODE:-direct}"
REASON_MAX_WORDS="${REASON_MAX_WORDS:-20}"

if [ -n "${SPARTAN_OPS_DIR:-}" ]; then
    REPO_DIR="$SPARTAN_OPS_DIR"
else
    REPO_DIR="$PROJ_BASE/external/spartan-ops"
fi
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/env.sh"
source "$REPO_DIR/modules.sh"

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$(dirname "$OUT")"

mamba activate alm-kimi

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
snaps = sorted(glob.glob(os.path.join(hub, "models--moonshotai--Kimi-Audio-7B-Instruct", "snapshots", "*")))
if not snaps:
    sys.exit("ERROR: no Kimi-Audio snapshot — run setup/setup_kimi_audio.sh")
print(snaps[-1])
PY
)

echo "[$(date -Is)] RAVLT Kimi-Audio"
echo "  job_id:          ${SLURM_JOB_ID:-(none)}"
echo "  node:            $(hostname)"
echo "  manifest:        $MANIFEST"
echo "  output:          $OUT"
echo "  model:           $MODEL_REF"
echo "  max_items:       $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

python "$JACK_BENCH/kimi_official_benchmark.py" \
    --model-dir "$MODEL_REF" \
    --manifest "$MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --log-every 5 \
    --mode "$MODE" \
    --reason-max-words "$REASON_MAX_WORDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --retry-max-new-tokens 64 \
    --empty-retry 2 \
    --merge-gap-sec 0.25 \
    --output "$OUT"

echo "[$(date -Is)] done -> $OUT"

# sample_id passthrough sanity check
if ! python -c "
import json, sys
with open('$OUT') as f:
    first = json.loads(f.readline())
sid = first.get('sample_id')
if not sid:
    print('WARNING: first output row has no sample_id', file=sys.stderr); sys.exit(2)
print(f'first sample_id: {sid}')
"; then
    echo "WARNING: predictions JSONL missing sample_id."
fi
