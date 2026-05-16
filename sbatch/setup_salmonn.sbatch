#!/bin/bash
# Background bootstrap of alm-salmonn env + SALMONN-13B + Vicuna-13B + Whisper + BEATs.
# Submit from project root on Spartan:
#   sbatch external/spartan-ops/sbatch/setup_salmonn.sbatch

#SBATCH -J setup_salmonn
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/kaip1/slurm_logs/%x-%j.err

set -euo pipefail

PROJ_BASE="/data/gpfs/projects/punim2341/kaip1/audio-cognition-benchmark"
REPO_DIR="$PROJ_BASE/external/spartan-ops"
[ -f "$REPO_DIR/env.sh" ] || { echo "ERROR: env.sh not under $REPO_DIR" >&2; exit 1; }

# setup_salmonn.sh refuses to run on login nodes via hostname check; on
# compute nodes (e.g. spartan-bm006) it proceeds normally.
bash "$REPO_DIR/setup/setup_salmonn.sh"
