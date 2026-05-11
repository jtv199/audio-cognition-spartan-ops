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

Each model needs two files:
1. `setup/setup_<model>.sh` — env creation + deps install + weight download.
2. `sbatch/run_<model>.sbatch` — job submission that activates the env and runs the matching `external/jack_benchmark/<model>_official_benchmark.py` runner.

Use `setup_kimi_audio.sh` and `run_kimi_audio.sbatch` as templates; vary only the env name, the pip install line, the HF model ID, and the runner path.
