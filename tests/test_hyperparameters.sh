#!/bin/bash
# Test different model hyperparameters

DATA="data_analysis/data/vtt-q50-28012026.csv"
OUTPUT_DIR="plots/experiments"

echo "Testing different model configurations..."
echo "========================================="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Test different degrees and alphas
for degree in 2 3; do
  for alpha in 0.001 0.01 0.1; do
    exp_name="deg${degree}_alpha${alpha}"
    exp_dir="$OUTPUT_DIR/$exp_name"

    echo "Testing degree=$degree, alpha=$alpha"
    echo "Output: $exp_dir"

    uv run resource-estimator-validate \
      --data $DATA \
      --degree $degree \
      --alpha $alpha \
      --max-error 30 \
      --plots $exp_dir \
      2>&1 | tee $exp_dir/results.log

    # Extract key metrics
    echo "  Results for $exp_name:" >> $OUTPUT_DIR/summary.txt
    grep -E "(R² Score|Mean error|Median error|Max error)" $exp_dir/results.log >> $OUTPUT_DIR/summary.txt
    echo "" >> $OUTPUT_DIR/summary.txt

    echo "  ✓ Complete"
    echo ""
  done
done

echo "========================================="
echo "All experiments complete!"
echo ""
echo "Summary of results:"
cat $OUTPUT_DIR/summary.txt
echo ""
echo "Detailed plots available in: $OUTPUT_DIR/"
