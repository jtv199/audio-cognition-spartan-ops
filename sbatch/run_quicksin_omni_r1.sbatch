#!/bin/bash
# Run Omni-R1 (Haoz0206/Omni-R1) on the QuickSIN-style manifest.
#
# Delegates to run_omni_r1.sbatch via exec after setting env vars.
# Mirrors run_hint_omni_r1.sbatch but with QuickSIN-tuned defaults.
#
# SUBMIT (from project root on Spartan):
#   sbatch external/spartan-ops/sbatch/run_quicksin_omni_r1.sbatch \
#       $PWD/pc-benchmark-pilot/manifests/quicksin_manifest.jsonl

#SBATCH -J omni_r1_quicksin
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

MANIFEST="${1:-${MANIFEST:?usage: sbatch run_quicksin_omni_r1.sbatch <manifest.jsonl>}}"
export MANIFEST
export OUT="${OUT:-$PWD/runs/omni_r1_quicksin.raw.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export MAX_ITEMS="${MAX_ITEMS:-0}"

# Resolve sibling sbatch path via SLURM_SUBMIT_DIR.
# (BASH_SOURCE[0] under slurm resolves to /var/spool/slurm/job<id>/ — see commit
#  22fe496 in spartan-ops for the rationale behind this pattern.)
_here="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/external/spartan-ops/sbatch"
exec bash "$_here/run_omni_r1.sbatch"
