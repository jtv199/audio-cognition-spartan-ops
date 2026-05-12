#!/bin/bash
# Run Omni-R1 (Haoz0206/Omni-R1) against a manifest on Spartan.
#
# SUBMIT:
#   MANIFEST=$PWD/pc-benchmark-pilot/manifests/<name>.jsonl \
#   OUT=$PWD/runs/omni_r1_<name>.raw.jsonl \
#   sbatch external/spartan-ops/sbatch/run_omni_r1.sbatch
#
# Prerequisites: setup/setup_omni_r1.sh has been run successfully.

#SBATCH -J omni_r1
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.jsonl}"
OUT="${OUT:?set OUT=/path/to/output.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

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
    print("Run setup/setup_omni_r1.sh first.", file=sys.stderr)
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

echo "[$(date -Is)] starting Omni-R1 inference"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  model:     $MODEL_REF"
echo "  base:      $BASE_REF"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

# omni_r1_local_smoke.py exposes run_smoke(model_dir, base_model_path, prompt,
# audio_paths, ...). Drive it across a manifest with this inline loop and emit
# jack_benchmark-style output rows so predictions_to_replay.py can ingest them
# the same way it does AF3 / Phi-4-MM / Kimi runs.

python - "$MANIFEST" "$OUT" "$MODEL_REF" "$BASE_REF" "$MAX_ITEMS" "$MAX_NEW_TOKENS" <<'PY'
import json, os, socket, sys, time
sys.path.insert(0, os.environ["JACK_BENCH"])

# qwen_omni_utils >=0.0.4 made process_mm_info return a 3-tuple by default,
# only returning the 4-tuple form when return_video_kwargs=True. The runner
# (omni_r1_local_smoke.py:242) unpacks 4 values unconditionally. Monkey-patch
# the function to always pass return_video_kwargs=True so the unpack works.
import qwen_omni_utils  # noqa: E402
_orig_process_mm_info = qwen_omni_utils.process_mm_info
def _patched_process_mm_info(*args, **kwargs):
    kwargs.setdefault("return_video_kwargs", True)
    return _orig_process_mm_info(*args, **kwargs)
qwen_omni_utils.process_mm_info = _patched_process_mm_info

from omni_r1_local_smoke import run_smoke  # noqa: E402
# After omni_r1_local_smoke imports process_mm_info into its own namespace at
# call time (inside run_smoke), we need to patch the imported reference too:
import omni_r1_local_smoke as _orls  # noqa: E402
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
        allowed = [str(x).strip() for x in item.get("allowed_outputs", []) if str(x).strip()]
        if allowed and len(allowed) <= 16:
            prompt = f"{prompt}\nOutput exactly one label from: {', '.join(allowed)}"
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
            "task_name": str(item.get("task_name", "")),
            "selection_tier": str(item.get("selection_tier", "")),
            "benchmark_prompt": str(item.get("prompt", "")),
            "rendered_prompt": prompt,
            "allowed_outputs": allowed,
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
