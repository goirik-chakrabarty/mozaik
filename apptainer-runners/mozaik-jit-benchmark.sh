#!/bin/bash

# Mozaik JIT Benchmark Runner
# This script runs the JIT benchmark within an apptainer container
# and produces timing and equivalence reports

set -e

echo "=== Mozaik JIT Benchmark ==="
echo "CWD: $(pwd)"

# Check if benchmark_jit.py exists
if [ ! -f "/mozaik/benchmark_jit.py" ]; then
    echo "Error: benchmark_jit.py not found in /mozaik/"
    exit 1
fi

# Default values
MODEL=${1:-"devtools.dummy_model.DummyModel"}
CONFIG=${2:-"param/defaults"}
OUTPUT_DIR=${3:-"/tmp/mozaik_benchmark_results"}
DURATION=${4:-1000}

echo "Model: $MODEL"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Duration: ${DURATION}ms"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
python /mozaik/benchmark_jit.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    --duration "$DURATION"

# Report path
REPORT="$OUTPUT_DIR/benchmark_report.json"

if [ -f "$REPORT" ]; then
    echo ""
    echo "=== Benchmark Report ==="
    python3 -m json.tool "$REPORT"
    echo "Report saved to: $REPORT"
else
    echo "Warning: Benchmark report not found at $REPORT"
    exit 1
fi

exit 0
