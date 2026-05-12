#!/bin/bash
# Run Step-Audio-2-mini against a manifest on Spartan.
#
# SUBMIT:
#   MANIFEST=$PWD/pc-benchmark-pilot/manifests/<name>.jsonl \
#   OUT=$PWD/runs/step_audio2_<name>.raw.jsonl \
#   sbatch external/spartan-ops/sbatch/run_step_audio2_mini.sbatch
#
# Prerequisites: setup/setup_step_audio2_mini.sh has been run successfully.

#SBATCH -J step_audio2
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

mamba activate alm-step-audio2

# StepAudio2 expects the model identifier as a path relative to its working
# directory. cd into Step-Audio2 (with the model symlinked at ./Step-Audio-2-mini)
# before invoking inference.
STEP_REPO="$PROJ/Step-Audio2"
cd "$STEP_REPO"

echo "[$(date -Is)] starting Step-Audio-2-mini inference"
echo "  job_id:    ${SLURM_JOB_ID:-(none)}"
echo "  node:      $(hostname)"
echo "  manifest:  $MANIFEST"
echo "  output:    $OUT"
echo "  cwd:       $PWD"
echo "  max_items: $MAX_ITEMS  max_new_tokens: $MAX_NEW_TOKENS"
nvidia-smi -L 2>&1 | head -4

# Inline manifest driver. Concatenates multi-audio clips with a 0.25s gap
# (same pattern as AF3 / Phi-4-MM / Omni-R1), feeds the merged WAV plus the
# question prompt to StepAudio2, captures the model's text output.

python - "$MANIFEST" "$OUT" "$STEP_REPO" "$MAX_ITEMS" "$MAX_NEW_TOKENS" <<'PY'
import json, os, socket, sys, tempfile, time, traceback
import numpy as np
import soundfile as sf

manifest_path, out_path, step_repo, max_items_s, max_new_tokens_s = sys.argv[1:6]
max_items = int(max_items_s)
max_new_tokens = int(max_new_tokens_s)

# stepaudio2 lives in $STEP_REPO; we cd'd there in the sbatch already so the
# package's relative-path assumptions hold (it loads sub-modules via local
# resolution).
sys.path.insert(0, step_repo)
from stepaudio2 import StepAudio2  # noqa: E402

print("[init] loading Step-Audio-2-mini", flush=True)
t0 = time.time()
model = StepAudio2("Step-Audio-2-mini")
print(f"[init] loaded in {time.time()-t0:.1f}s", flush=True)

TARGET_SR = 16000
GAP_SEC = 0.25
SYS_PROMPT = "You are an audio assistant. Listen carefully and answer the user's question concisely."


def _load_mono(path, target_sr=TARGET_SR):
    audio, sr = sf.read(path, always_2d=False)
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if int(sr) != target_sr:
        x = np.arange(audio.shape[0], dtype=np.float32)
        new_len = max(1, int(round(audio.shape[0] * target_sr / sr)))
        x_new = np.linspace(0, audio.shape[0] - 1, new_len, dtype=np.float32)
        audio = np.interp(x_new, x, audio).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _concat_audio(paths, gap_sec=GAP_SEC):
    clips = [_load_mono(p) for p in paths if os.path.exists(p)]
    if not clips:
        return None, 0
    if len(clips) == 1:
        return clips[0], 1
    gap = np.zeros((int(gap_sec * TARGET_SR),), dtype=np.float32)
    pieces = []
    for i, c in enumerate(clips):
        pieces.append(c)
        if i < len(clips) - 1:
            pieces.append(gap)
    return np.concatenate(pieces), len(clips)


def _build_prompt(item):
    base = str(item.get("prompt", "")).strip() or "Describe this audio briefly."
    allowed = [str(x).strip() for x in item.get("allowed_outputs", []) if str(x).strip()]
    if allowed and len(allowed) <= 16:
        base = f"{base}\nOutput exactly one label from: {', '.join(allowed)}"
    return base


host = socket.gethostname()
ok_count = 0
total = 0
with open(manifest_path, "r", encoding="utf-8") as rf, open(out_path, "w", encoding="utf-8") as wf:
    rows = [json.loads(ln) for ln in rf if ln.strip()]
    if max_items > 0:
        rows = rows[:max_items]
    total = len(rows)
    for idx, item in enumerate(rows, start=1):
        prompt = _build_prompt(item)
        audio_paths = [str(p) for p in item.get("audio_paths", []) if str(p).strip()]
        row = {
            "model": "step-audio-2-mini",
            "run_index": idx,
            "runs_total": total,
            "item_id": str(item.get("item_id", "")),
            "sample_id": str(item.get("sample_id", item.get("item_id", ""))),
            "task_name": str(item.get("task_name", "")),
            "selection_tier": str(item.get("selection_tier", "")),
            "benchmark_prompt": str(item.get("prompt", "")),
            "rendered_prompt": prompt,
            "allowed_outputs": [str(x).strip() for x in item.get("allowed_outputs", []) if str(x).strip()],
            "answer": str(item.get("answer", "")).strip(),
            "audio_paths": audio_paths,
            "audio_path_count": len(audio_paths),
            "host": host,
            "timestamp": int(time.time()),
            "ok": False,
            "mode": "step_audio2_concat_generate",
        }
        tmp_wav = None
        try:
            merged, clip_count = _concat_audio(audio_paths)
            if merged is None or merged.size == 0:
                row["error"] = "no_audio_paths_or_missing_files"
                wf.write(json.dumps(row, ensure_ascii=False) + "\n"); wf.flush()
                continue
            # Write the merged audio to a temp WAV that StepAudio2 can ingest.
            tmp_wav = tempfile.NamedTemporaryFile(prefix="step_audio2_", suffix=".wav", delete=False)
            tmp_wav.close()
            sf.write(tmp_wav.name, merged, TARGET_SR)
            row["merged_audio_clip_count"] = clip_count
            row["merged_audio_seconds"] = float(merged.shape[0] / TARGET_SR)

            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "human", "content": [
                    {"type": "audio", "audio": tmp_wav.name},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": None},
            ]
            tokens, text, audio_out = model(
                messages, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False
            )
            decoded = str(text or "").strip() or "[empty_generation]"
            row["ok"] = True
            row["response_text"] = decoded
            row["response_preview"] = decoded[:240]
        except Exception as e:
            row["error"] = repr(e)
            row["trace"] = traceback.format_exc(limit=8)
        finally:
            if tmp_wav and os.path.exists(tmp_wav.name):
                try: os.remove(tmp_wav.name)
                except Exception: pass
        wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        wf.flush()
        if row.get("ok"):
            ok_count += 1
        print(json.dumps({"run_index": idx, "total": total, "item_id": row["item_id"],
                          "ok": row["ok"],
                          "response_head": str(row.get("response_text", ""))[:80]},
                         ensure_ascii=False), flush=True)
print(json.dumps({"output": out_path, "total": total, "ok": ok_count,
                  "bad": total - ok_count, "manifest": manifest_path}, ensure_ascii=False))
PY

echo "[$(date -Is)] finished — wrote $OUT"
