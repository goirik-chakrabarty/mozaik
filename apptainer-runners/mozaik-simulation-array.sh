cd /project

# 1. Set default offset if not provided (safe fallback)
if [ -z "$STIM_OFFSET" ]; then
    STIM_OFFSET=0
fi

# 2. Create a unique run name using the offset
RUN_NAME="test:fullbig32_${STIM_OFFSET}"

# 3. Construct the expected output directory name
# Mozaik typically constructs this as: ModelName_RunName_____
DIR_NAME="SelfSustainedPushPull_${RUN_NAME}_____"

echo "Running simulation with name: $RUN_NAME"
echo "Cleaning up directory: $DIR_NAME"

# 4. Remove the specific directory for THIS job only
rm -rf "$DIR_NAME"

# 5. Run the python script passing the UNIQUE Run Name
python -u run.py nest 32 param_MSA/defaults "$RUN_NAME"