#!/bin/bash
# Run Omni-R1 on the full CST-48 manifest.
#
# Thin wrapper around run_omni_r1.sbatch with CST-tuned MAX_NEW_TOKENS.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_cst_48_omni_r1.sbatch \
#       $PWD/pc-benchmark-pilot/manifests/cst_48_manifest.jsonl

#SBATCH -J omni_r1_cst_48
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

MANIFEST="${1:-${MANIFEST:?usage: sbatch run_cst_48_omni_r1.sbatch <manifest.jsonl>}}"
export MANIFEST
export OUT="${OUT:-$PWD/runs/omni_r1_cst_48.raw.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
export MAX_ITEMS="${MAX_ITEMS:-0}"

# Resolve sibling sbatch path via SLURM_SUBMIT_DIR — BASH_SOURCE[0] under slurm
# resolves to /var/spool/slurm/job<id>/ which only contains the launcher copy.
_here="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/external/spartan-ops/sbatch"
# Re-exec the canonical Omni-R1 runner with the env we set above.
exec bash "$_here/run_omni_r1.sbatch"
