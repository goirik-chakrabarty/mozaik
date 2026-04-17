#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=mozaik-test33
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --array=0-19
#SBATCH --time=10:00:00
#SBATCH --output=./slurm_test33/mozaik-test33-%A_%a.out
#SBATCH --error=./slurm_test33/mozaik-test33-%A_%a.err
#SBATCH -p medium96s
#SBATCH -C sapphirerapids
#SBATCH --mail-type=ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

# Threading environment
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE

# 20 trials, 1 chunk each — array ID = trial
N_CHUNKS=1
export TRIAL=$SLURM_ARRAY_TASK_ID
export CHUNK=0
export CHUNK_DIR="/data/mozaik_chunk_test33"

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Trial: ${TRIAL}, Chunk: ${CHUNK}"
echo "Chunk dir: ${CHUNK_DIR}"

echo "Submitting from: ${SLURM_SUBMIT_DIR}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

mkdir -p ./slurm_test33

bash apptainer-compose-array.sh
