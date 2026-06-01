#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=mozaik-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=150G
#SBATCH --array=1
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_export_dynamic26872-17-20/export-%A_%a.out
#SBATCH --error=./slurm_export_dynamic26872-17-20/export-%A_%a.err
#SBATCH -p standard96
#SBATCH -C cascadelake # sapphirerapids or cascadelake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

# Thread pools for numpy/scipy — no MPI, no NEST
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_DYNAMIC=FALSE

# Each array slot handles exactly one trial
export TRIAL=$SLURM_ARRAY_TASK_ID

mkdir -p ./slurm_export_dynamic26872-17-20

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Exporting trial: ${TRIAL}"
echo "Submitting from: ${SLURM_SUBMIT_DIR}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

export CHUNK_START=0 CHUNK_END=12 BATCH_SIZE=12 EXPORT_MODE="--screen-only"
bash cluster/apptainer-compose-export.sh
