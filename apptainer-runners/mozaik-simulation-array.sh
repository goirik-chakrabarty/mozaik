cd /project

# 1. Set defaults if not provided (safe fallback)
TRIAL=${TRIAL:-0}
CHUNK=${CHUNK:-0}

# 2. Create a unique run name using trial and chunk
RUN_NAME="trial${TRIAL}_chunk${CHUNK}"

# 3. Construct the expected output directory name
# Mozaik typically constructs this as: ModelName_RunName_____
DIR_NAME="SelfSustainedPushPull_${RUN_NAME}_____"

echo "Running simulation with name: $RUN_NAME"
echo "Cleaning up directory: $DIR_NAME"

# 4. Remove the specific directory for THIS job only
rm -rf "$DIR_NAME"

echo "--- Starting Simulation (Internal MPI) ---"
echo "Host Thread Limit Check: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Running with $NTASKS MPI Tasks"
echo "Trial: $TRIAL, Chunk: $CHUNK"

# 5. Compute per-trial mozaik_seed so each trial has independent noise
#    but identical network topology.
#    - pynn_seed (fixed in param/defaults=5): controls connectivity, positions, weights
#    - mozaik_seed (varied here): controls background noise generators via mozaik.get_seeds()
SEED_OFFSET=$(( TRIAL * 1000 + CHUNK ))
MOZAIK_SEED=$(( 1023 + SEED_OFFSET ))
echo "Per-trial seed: mozaik_seed=$MOZAIK_SEED (pynn_seed=5 fixed, offset=$SEED_OFFSET)"

# 6. Run the python script passing the UNIQUE Run Name
#    Modified parameters (key value pairs) go between param_file and run_name.
mpirun \
    -n $NTASKS \
    -x OMP_NUM_THREADS \
    -x MKL_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x PYTHONPATH \
    python -u run.py nest $NTASKS param/defaults mozaik_seed $MOZAIK_SEED "$RUN_NAME"