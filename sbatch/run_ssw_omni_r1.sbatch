#!/bin/bash
# Run Omni-R1 on the SSW (Staggered Spondaic Word) manifest.
# Mirrors run_ssi_icm_omni_r1.sbatch.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_ssw_omni_r1.sbatch \
#       $PWD/pc-benchmark-pilot/manifests/ssw_manifest.jsonl

#SBATCH -J omni_r1_ssw
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

MANIFEST="${1:-${MANIFEST:?usage: sbatch run_ssw_omni_r1.sbatch <manifest.jsonl>}}"
export MANIFEST
export OUT="${OUT:-$PWD/runs/omni_r1_ssw.raw.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export MAX_ITEMS="${MAX_ITEMS:-0}"

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

mkdir -p "$XDG_CACHE_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$HOME" "$TMPDIR" \
         "$(dirname "$OUT")"

mamba activate alm-omni-r1

MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--Haoz0206--Omni-R1", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    print(f"ERROR: no Omni-R1 snapshot under {pat}", file=sys.stderr)
    sys.exit(1)
print(snaps[-1])
PY
)
BASE_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--Qwen--Qwen2.5-Omni-7B", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    print(f"ERROR: no Qwen2.5-Omni-7B snapshot under {pat}", file=sys.stderr)
    sys.exit(1)
print(snaps[-1])
PY
)

echo "[$(date -Is)] starting Omni-R1 SSW inference"
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
_orig_process_mm_info = qwen_omni_utils.process_mm_info
def _patched_process_mm_info(*args, **kwargs):
    kwargs.setdefault("return_video_kwargs", True)
    return _orig_process_mm_info(*args, **kwargs)
qwen_omni_utils.process_mm_info = _patched_process_mm_info

from omni_r1_local_smoke import run_smoke
import omni_r1_local_smoke as _orls
if hasattr(_orls, "process_mm_info"):
    _orls.process_mm_info = _patched_process_mm_info

manifest_path, out_path, model_ref, base_ref, max_items_s, max_new_tokens_s = sys.argv[1:7]
max_items = int(max_items_s)
max_new_tokens = int(max_new_tokens_s)
os.environ["AUDIOCOG_MAX_NEW_TOKENS"] = str(max_new_tokens)

host = socket.gethostname()
ok_count = 0
total = 0
with open(manifest_path, "r", encoding="utf-8") as rf, open(out_path, "w", encoding="utf-8") as wf:
    rows = [json.loads(ln) for ln in rf if ln.strip()]
    if max_items > 0:
        rows = rows[:max_items]
    total = len(rows)
    for idx, item in enumerate(rows, start=1):
        prompt = str(item.get("prompt", "")).strip() or "Describe this audio briefly."
        audio_paths = [str(p) for p in item.get("audio_paths", []) if str(p).strip()]
        result = run_smoke(
            model_dir=model_ref,
            base_model_path=base_ref,
            prompt=prompt,
            audio_paths=audio_paths,
            max_new_tokens=max_new_tokens,
        )
        decoded = str(result.get("response_text") or "").strip()
        if not decoded:
            decoded = "[empty_generation]"
        row = {
            "model": "omni-r1",
            "run_index": idx,
            "runs_total": total,
            "item_id": str(item.get("item_id", "")),
            "sample_id": str(item.get("sample_id", item.get("item_id", ""))),
            "task_name": "ssw",
            "benchmark_prompt": str(item.get("prompt", "")),
            "rendered_prompt": prompt,
            "allowed_outputs": item.get("allowed_outputs", []),
            "answer": str(item.get("answer", "")).strip(),
            "audio_paths": audio_paths,
            "audio_path_count": len(audio_paths),
            "host": host,
            "timestamp": int(time.time()),
            "ok": bool(result.get("ok")),
            "mode": result.get("mode", ""),
            "audio_source": result.get("audio_source", ""),
            "merged_audio": bool(result.get("merged_audio", False)),
            "response_text": decoded,
            "response_preview": decoded[:240],
            "metadata": item.get("metadata", {}),
        }
        if result.get("error"):
            row["error"] = result.get("error")
        wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        wf.flush()
        if result.get("ok"):
            ok_count += 1
        print(json.dumps({"run_index": idx, "total": total, "item_id": row["item_id"],
                          "ok": row["ok"], "response_head": decoded[:80]}, ensure_ascii=False))
print(json.dumps({"output": out_path, "total": total, "ok": ok_count,
                  "bad": total - ok_count, "manifest": manifest_path}, ensure_ascii=False))
PY

echo "[$(date -Is)] finished — wrote $OUT"
