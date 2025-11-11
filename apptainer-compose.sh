#!/bin/bash

# Define variables

# PROJECT_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/mozaik-models/Rozsa_Cagnol2024" 
PROJECT_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/mozaik-models/experanto" 
SIF_IMAGE="$PWD/mozaik.sif"
ENV_FILE=".env"
MOZAIK_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/mozaik"
DATA_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/data"

echo PROJECT_ROOT: $PROJECT_ROOT        
echo SIF_IMAGE: $SIF_IMAGE
echo ENV_FILE: $ENV_FILE
echo MOZAIK_ROOT: $MOZAIK_ROOT

# # Export environment variables from .env file
# set -o allexport
# source "$ENV_FILE"
# set +o allexport

# Run the container
apptainer exec \
 --cleanenv \
 --env OMPI_MCA_orte_tmpdir_base=/tmp \
 --env PYTHONPATH="/mozaik:$PYTHONPATH" \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$MOZAIK_ROOT:/mozaik" \
 --bind "$DATA_ROOT:/data" \
 "$SIF_IMAGE" \
 bash apptainer-runners/mozaik-model.sh