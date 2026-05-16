#!/bin/bash
# Run Omni-R1 on the ACPT manifest (Keith 1994).
#
# 20 trials × ~3.3 min single-syllable word stream; subject counts 'dog' targets.
# MAX_NEW_TOKENS=32 — answer is a single integer.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_acpt_omni_r1.sbatch

#SBATCH -J acpt_omni_r1
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

PROJ_BASE="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
MANIFEST="${MANIFEST:-${1:-$PROJ_BASE/pc-benchmark-pilot/manifests/acpt_manifest.jsonl}}"
OUT="${OUT:-$PROJ_BASE/runs/omni_r1_acpt.raw.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

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

mamba activate alm-omni-r1

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--Haoz0206--Omni-R1", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    sys.exit("ERROR: no Omni-R1 snapshot — run setup/setup_omni_r1.sh")
print(snaps[-1])
PY
)
BASE_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--Qwen--Qwen2.5-Omni-7B", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    sys.exit("ERROR: no Qwen2.5-Omni-7B snapshot")
print(snaps[-1])
PY
)

echo "[$(date -Is)] ACPT Omni-R1"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  model:     $MODEL_REF"
echo "  base:      $BASE_REF"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

python - "$MANIFEST" "$OUT" "$MODEL_REF" "$BASE_REF" "$MAX_ITEMS" "$MAX_NEW_TOKENS" <<'PY'
import json, os, socket, sys, time
sys.path.insert(0, os.environ["JACK_BENCH"])

import qwen_omni_utils
_orig = qwen_omni_utils.process_mm_info
def _p(*a, **k):
    k.setdefault("return_video_kwargs", True)
    return _orig(*a, **k)
qwen_omni_utils.process_mm_info = _p

from omni_r1_local_smoke import run_smoke
import omni_r1_local_smoke as _orls
if hasattr(_orls, "process_mm_info"):
    _orls.process_mm_info = _p

manifest_path, out_path, model_ref, base_ref, max_items_s, max_new_tokens_s = sys.argv[1:7]
max_items = int(max_items_s)
max_new_tokens = int(max_new_tokens_s)
host = socket.gethostname()
ok_count = 0

DEFAULT_PROMPT = (
    "Listen to the stream of words. "
    "Count how many times you heard the word 'dog'. "
    "Answer with a single integer."
)

with open(manifest_path, encoding="utf-8") as rf, open(out_path, "w", encoding="utf-8") as wf:
    rows = [json.loads(ln) for ln in rf if ln.strip()]
    if max_items > 0:
        rows = rows[:max_items]
    total = len(rows)
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {total} items -> {out_path}")
    for idx, item in enumerate(rows, 1):
        prompt = str(item.get("prompt", "")).strip() or DEFAULT_PROMPT
        audio_paths = [str(p) for p in item.get("audio_paths", []) if str(p).strip()]
        try:
            result = run_smoke(model_ref, base_ref, prompt, audio_paths, max_new_tokens=max_new_tokens)
            decoded = str(result.get("response_text") or "").strip() or "[empty_generation]"
            ok = bool(result.get("ok"))
        except Exception as e:
            decoded = f"[error: {e!r}]"
            ok = False
        row = {
            "model": "omni-r1",
            "run_index": idx,
            "runs_total": total,
            "item_id": str(item.get("item_id", "")),
            "sample_id": str(item.get("sample_id", item.get("item_id", ""))),
            "task_name": "acpt",
            "benchmark_prompt": prompt,
            "answer": str(item.get("answer", "")).strip(),
            "audio_paths": audio_paths,
            "host": host,
            "timestamp": int(time.time()),
            "ok": ok,
            "response_text": decoded,
            "response_preview": decoded[:120],
            "metadata": item.get("metadata", {}),
        }
        wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        wf.flush()
        if ok:
            ok_count += 1
        if idx % 2 == 0 or idx == total:
            print(json.dumps({"idx": idx, "total": total, "item_id": row["item_id"],
                              "ok": ok, "response_head": decoded[:80]}, ensure_ascii=False))

print(json.dumps({"output": out_path, "total": total, "ok": ok_count,
                  "bad": total - ok_count}, ensure_ascii=False))
PY

echo "[$(date -Is)] done -> $OUT"
