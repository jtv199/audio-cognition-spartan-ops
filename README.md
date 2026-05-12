# audio-cognition-spartan-ops

Spartan HPC operations for [audio-cognition-benchmark](https://github.com/jtv199/audio-cognition-benchmark).

## What's here

- `env.sh` — shared environment variables (project root, HF cache, conda paths). Source first.
- `modules.sh` — the Spartan-standard toolchain `module load` block (foss/2022a + CUDA/12.4.1 + UCX-CUDA + cuDNN + Mambaforge). Source after `env.sh`.
- `setup/setup_<model>.sh` — one-time bootstrap per model: creates the mamba env, installs model-specific deps, pre-downloads weights. Run inside `sinteractive -p interactive` + `tmux`.
- `sbatch/run_<model>.sbatch` — Slurm job per model: loads modules, activates env, runs the matching `external/jack_benchmark/<model>_official_benchmark.py` runner against a manifest, writes predictions JSONL.

## Account / project context

| | Value |
|---|---|
| **Project** | `punim2341` |
| **User** | `kaip1` |
| **Work dir** | `/data/gpfs/projects/punim2341/kaip1/` |
| **Partition** | `gpu-a100-short` (4h cap) or `gpu-a100` (7d) |
| **QOS** | `normal` (default for `punim2341`) |
| **Mamba envs go to** | `$PROJ/envs/<model-name>/` |
| **HF cache** | `$PROJ/hf_cache/` |

**NOT applicable here**: `feit-gpu-a100` partition / `--qos=feit`. That binding works for the `punim2758` team (Hongyu/Austin/Siyi) but not for `punim2341`. Stay on the public partitions.

## Workflow

```bash
# 1. One-time, per model (~30-60 min, mostly the HF download)
ssh spartan
sinteractive -p interactive --time=04:00:00 -c 4 --mem=16G
tmux new -s setup
cd /data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark
bash external/spartan-ops/setup/setup_kimi_audio.sh

# 2. Build a manifest from your Inspect dataset (laptop or Spartan, both work)
python tools/manifest_from_inspect.py \
    pc-benchmark-pilot/datasets/timit_single_phoneme_15.jsonl \
    --out pc-benchmark-pilot/manifests/timit_single_phoneme_15.manifest.jsonl

# 3. Submit the inference job
MANIFEST=$PWD/pc-benchmark-pilot/manifests/timit_single_phoneme_15.manifest.jsonl \
OUT=$PWD/runs/kimi_timit_single_phoneme_15_raw.jsonl \
sbatch external/spartan-ops/sbatch/run_kimi_audio.sbatch

# 4. Watch
squeue -u $USER
tail -f /data/gpfs/projects/punim2341/kaip1/slurm_logs/kimi_audio_timit-*.out

# 5. When done, convert + replay locally via Inspect (laptop side)
python tools/predictions_to_replay.py \
    runs/kimi_timit_single_phoneme_15_raw.jsonl \
    --out logs/predictions_kimi_timit_single_phoneme_15.jsonl
python -m inspect_ai eval eval_timit_15.py@timit_single_phoneme \
    --model replay/kimi-audio \
    -M predictions=logs/predictions_kimi_timit_single_phoneme_15.jsonl \
    --log-dir logs/kimi_single_phoneme_15
```

## Login-node safety

Both `setup/*.sh` and `sbatch/*.sbatch` refuse to run on a login node:

```bash
case "$(hostname)" in
    *login*) echo "ERROR: do not run on a login node"; exit 1 ;;
esac
```

`sinteractive` first. Always.

## Inode discipline

The project shares 7M inodes across all users. Every mamba env is ~150-250k files, every HF model cache is ~50k files. Be deliberate:
- Don't keep dead envs around. `mamba env remove -n <name>` when done with a model.
- Pack large reference data as tar archives where possible.
- Periodic `find $PROJ/kaip1 -type f | wc -l` to spot-check own usage.
- `check_project_usage punim2341` for the whole project.

## Adding a new model

Each model needs three files:
1. `setup/setup_<model>.sh` — env creation + deps install + weight download.
2. `setup/verify_<model>_env.sh` — env sanity check (torch / transformers / model classes / weights present).
3. `sbatch/run_<model>.sbatch` — job submission that activates the env and runs the matching `external/jack_benchmark/<model>_official_benchmark.py` runner.

Use `setup_kimi_audio.sh`, `verify_kimi_env.sh` and `run_kimi_audio.sbatch` as templates; vary only the env name, the pip install line, the HF model ID, and the runner path.

## Models installed

| Model | Env name | HF repo | Runner | Notes |
|---|---|---|---|---|
| Kimi-Audio-7B-Instruct | `alm-kimi` | `moonshotai/Kimi-Audio-7B-Instruct` | `kimi_official_benchmark.py` | `transformers>=4.45,<5.0`, requires editable install of `kimi-audio` repo for `kimia_infer` |
| Qwen2-Audio-7B-Instruct | `alm-qwen2-audio` | `Qwen/Qwen2-Audio-7B-Instruct` | `qwen2_audio_instruct_hf_smoke.py` (driven via inline loop in sbatch) | `transformers>=4.45,<5.0` |
| Audio Flamingo 3 | `alm-af3` | `nvidia/audio-flamingo-3-hf` | `af3_official_benchmark.py` | Use the `-hf` HF-format port (NOT `nvidia/audio-flamingo-3`, which is NVIDIA's LlavaLlama-format release for their custom codebase). `transformers` from `git+main` until `AudioFlamingo3ForConditionalGeneration` lands in a tagged release. NVIDIA OneWay Noncommercial license. |
| Phi-4-Multimodal-Instruct | `alm-phi4mm` | `microsoft/Phi-4-multimodal-instruct` | `phi4mm_official_benchmark.py` | Strict pin: `torch==2.6.0`, `transformers==4.48.2`, `peft==0.13.2`, `flash_attn==2.7.4.post1` |
| Omni-R1 | `alm-omni-r1` | `Haoz0206/Omni-R1` (fine-tuned) + `Qwen/Qwen2.5-Omni-7B` (processor) | `omni_r1_local_smoke.py` (driven via inline loop in sbatch) | RL-tuned Qwen2.5-Omni-Thinker; top MMAU Ga (54.5) from arXiv 2505.09439. Needs `qwen-omni-utils` pip pkg + dual-repo load. |

All four envs share `$HF_HOME` so common encoder weights (Whisper variants, etc.) dedupe.

## Smoke-test pattern (1-sample manifest)

The first row of `pc-benchmark-pilot/manifests/gudkar_wcgd_full30_manifest.jsonl` is the canonical smoke sample — it's a 3-clip WCGD audio with a multi-audio prompt, exercises the `audio_paths: List[str]` path that all four runners support.

```bash
# On Spartan, after submodule is up to date:
head -1 pc-benchmark-pilot/manifests/gudkar_wcgd_full30_manifest.jsonl \
    > pc-benchmark-pilot/manifests/smoke_1sample_manifest.jsonl

for model in qwen2_audio af3 phi4mm; do
    MANIFEST=$PWD/pc-benchmark-pilot/manifests/smoke_1sample_manifest.jsonl \
    OUT=$PWD/runs/smoke_${model}.raw.jsonl \
    sbatch external/spartan-ops/sbatch/run_${model}.sbatch
done
```

Pass criterion per model: output file has 1 row with `response_text` non-empty and not `[empty_generation]`.
