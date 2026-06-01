#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=export-test33
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --array=20-50
#SBATCH --time=06:00:00
#SBATCH --output=./slurm_test33/export-%A_%a.out
#SBATCH --error=./slurm_test33/export-%A_%a.err
#SBATCH -p standard96
#SBATCH -C cascadelake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_DYNAMIC=FALSE

export TRIAL=$SLURM_ARRAY_TASK_ID

mkdir -p ./slurm_test33

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Exporting trial: ${TRIAL}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

export CHUNK_START=0 CHUNK_END=1 BATCH_SIZE=1 N_CHUNKS=1
export CHUNK_DIR="/data/mozaik_chunk_test33"
export OUTPUT_PREFIX="/data/mozaik_data_test33/trial"

bash cluster/apptainer-compose-export.sh
