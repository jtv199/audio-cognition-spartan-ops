# Spartan-standard module load block, per the Austin/Dr-Hong-Jia project setup
# guide. Source with: `source "$REPO/external/spartan-ops/modules.sh"`.
#
# Order matters:
#   1. foss/2022a    — GCC + OpenMPI + numerical libs toolchain
#   2. CUDA/12.4.1   — GPU runtime
#   3. UCX-CUDA      — networking layer with CUDA support (required by NCCL ops
#                       in transformers/torch when scaling beyond single-GPU,
#                       but cheap to always load)
#   4. cuDNN         — required by torch GPU operations
#   5. Mambaforge    — python + conda/mamba env manager

source /etc/profile.d/modules.sh
module purge
module load foss/2022a
module load CUDA/12.4.1
module load UCX-CUDA/1.16.0-CUDA-12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1
module load Mambaforge/23.1.0

# Make `mamba activate` work non-interactively.
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(conda info --base)/etc/profile.d/mamba.sh" 2>/dev/null || true
