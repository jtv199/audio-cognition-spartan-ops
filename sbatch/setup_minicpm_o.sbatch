#!/bin/bash
# Background bootstrap of alm-minicpm-o env + MiniCPM-o-2_6 weights.
# Submit from project root on Spartan:
#   sbatch external/spartan-ops/sbatch/setup_minicpm_o.sbatch

#SBATCH -J setup_minicpm_o
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

PROJ_BASE="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
REPO_DIR="$PROJ_BASE/external/spartan-ops"
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }

# setup_minicpm_o.sh refuses to run on login nodes via hostname check; on
# compute nodes (e.g. spartan-bm006) it proceeds normally.
bash "$REPO_DIR/setup/setup_minicpm_o.sh"
