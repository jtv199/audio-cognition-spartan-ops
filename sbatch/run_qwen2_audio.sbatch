#!/bin/bash
# Run Qwen2-Audio-7B-Instruct against a manifest on Spartan.
#
# SUBMIT:
#   MANIFEST=$PWD/pc-benchmark-pilot/manifests/<name>.jsonl \
#   OUT=$PWD/runs/qwen2_audio_<name>.raw.jsonl \
#   sbatch external/spartan-ops/sbatch/run_qwen2_audio.sbatch
#
# Prerequisites: setup/setup_qwen2_audio.sh has been run successfully.

#SBATCH -J qwen2_audio
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

set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.jsonl}"
OUT="${OUT:?set OUT=/path/to/output.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

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

mamba activate alm-qwen2-audio

MODEL_REPO="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REF=$(python - <<PY
import glob, os, sys
hub = os.environ["HF_HUB_CACHE"]
pat = os.path.join(hub, "models--Qwen--Qwen2-Audio-7B-Instruct", "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    print(f"ERROR: no Qwen2-Audio snapshot under {pat}", file=sys.stderr)
    print("Run setup/setup_qwen2_audio.sh first.", file=sys.stderr)
    sys.exit(1)
print(snaps[-1])
PY
)

echo "[$(date -Is)] starting Qwen2-Audio inference"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  model:     $MODEL_REF"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

# The existing external/jack_benchmark/qwen2_audio_instruct_hf_smoke.py is a
# one-shot CLI smoke (single --prompt + --audio-path call). Drive it across a
# manifest with this inline loop and emit jack_benchmark-style output rows
# (item_id, task_name, response_text, audio_paths) so predictions_to_replay.py
# can ingest them identically to AF3 / Phi-4-MM runs.

python - "$MANIFEST" "$OUT" "$MODEL_REF" "$MAX_ITEMS" "$MAX_NEW_TOKENS" <<'PY'
import json, os, socket, sys, time
sys.path.insert(0, os.environ["JACK_BENCH"])
from qwen2_audio_instruct_hf_smoke import run  # noqa: E402

manifest_path, out_path, model_ref, max_items_s, max_new_tokens_s = sys.argv[1:6]
max_items = int(max_items_s)
os.environ["AUDIOCOG_MAX_NEW_TOKENS"] = str(int(max_new_tokens_s))

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
        result = run(model_ref, prompt, audio_paths)
        decoded = str(result.get("response_text") or "").strip()
        if not decoded:
            decoded = "[empty_generation]"
        row = {
            "model": "qwen2-audio-7b-instruct",
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
