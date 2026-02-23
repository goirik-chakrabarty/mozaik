#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=mozaik-optimized
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_medium/mozaik-opt-%j.out
#SBATCH --error=./slurm_medium/mozaik-opt-%j.err
#SBATCH -p standard96 # standard96 or medium96s
#SBATCH -C cascadelake # cascadelake or sapphirerapids
#SBATCH --mail-type=ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

# Threading environment
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# export NUMEXPR_NUM_THREADS=1 # Added
# export VECLIB_MAXIMUM_THREADS=1 # Added
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE

# Optional MPI/shared-memory tuning (verify OpenMPI version before enabling)
# export OMPI_MCA_btl_vader_eager_limit=65536
# export OMPI_MCA_btl_vader_rndv_eager_limit=65536
# export OMPI_MCA_btl=self,vader

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

# Printing out some info
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

# Run simulation with JIT DISABLED
echo ""
echo "=========================================="
echo "RUN 1: Simulation with JIT DISABLED"
echo "=========================================="
export MOZAIK_JIT=0
bash apptainer-compose-opt.sh
JIT_OFF_EXIT=$?

# Run simulation with JIT ENABLED
echo ""
echo "=========================================="
echo "RUN 2: Simulation with JIT ENABLED"
echo "=========================================="
export MOZAIK_JIT=1
bash apptainer-compose-opt.sh
JIT_ON_EXIT=$?

# Print summary
echo ""
echo "=========================================="
echo "EXECUTION SUMMARY"
echo "=========================================="
echo "JIT disabled exit code:  $JIT_OFF_EXIT"
echo "JIT enabled exit code:   $JIT_ON_EXIT"
if [ $JIT_OFF_EXIT -eq 0 ] && [ $JIT_ON_EXIT -eq 0 ]; then
    echo "✓ Both runs completed successfully"
else
    echo "⚠ One or more runs failed - check output above"
fi