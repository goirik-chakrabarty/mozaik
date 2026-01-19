#!/bin/bash

#SBATCH --job-name=32mozaik_simulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time 0-48:00
#SBATCH --constraint "inet" 
#SBATCH --mem=200G
#SBATCH --array=140                # the range can be any positive integers separated by hyphen
#SBATCH --output=./slurm_files/arraybig/slurm-%x-%A_%a.out  # Use %A_%a for array job logging
#SBATCH --error=./slurm_files/arraybig/slurm-%x-%A_%a.err
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

cd /mnt/vast-nhr/projects/nix00014/goirik/mozaik

# Calculate offset
STIMULUS_PER_JOB=8
STIM_OFFSET=$((SLURM_ARRAY_TASK_ID * STIMULUS_PER_JOB))
STIM_WINDOW=$STIMULUS_PER_JOB
export STIM_OFFSET
export STIM_WINDOW

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Stimulus per job: ${STIM_WINDOW}"
echo "Computed Offset: ${STIM_OFFSET}"

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

bash apptainer-compose-array.sh