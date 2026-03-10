#!/bin/bash

# Usage:
#   ./bench.sh [--test TEST_NAME] [--shape M,N,K] [--runs NUM_RUNS] [--kernel KERNEL_NAME] [--gpu GPU_ID]
#   ./bench.sh [--test TEST_NAME] [--shapes shapes.csv] [--runs NUM_RUNS] [--kernel KERNEL_NAME] [--gpu GPU_ID]
#
# Examples:
#   ./bench.sh --test test_dbuf_4wave_mxfp_asymmetric_gemm --shape 16384,57344,16384
#   ./bench.sh --test test_dbuf_4wave_mxfp_asymmetric_gemm --shapes shapes.csv --gpu 3
#
# CSV format (one shape per line, no header):
#   1024,1024,8192
#   16384,57344,16384

TEST="test_dbuf_4wave_mxfp_asymmetric_gemm"
SHAPE=""
SHAPES_CSV=""
RUNS=5
KERNEL="gemm"
GPU=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)    TEST="$2";       shift 2 ;;
        --shape)   SHAPE="$2";      shift 2 ;;
        --shapes)  SHAPES_CSV="$2"; shift 2 ;;
        --runs)    RUNS="$2";       shift 2 ;;
        --kernel)  KERNEL="$2";     shift 2 ;;
        --gpu)     GPU="$2";        shift 2 ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build the list of shapes to benchmark
shapes=()
if [[ -n "$SHAPES_CSV" ]]; then
    if [[ ! -f "$SHAPES_CSV" ]]; then
        echo "ERROR: CSV file '$SHAPES_CSV' not found."
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        line=$(echo "$line" | tr -d '[:space:]')
        [[ -z "$line" || "$line" == \#* ]] && continue
        shapes+=("$line")
    done < "$SHAPES_CSV"
elif [[ -n "$SHAPE" ]]; then
    shapes+=("$SHAPE")
else
    shapes+=("1024,1024,8192")
fi

run_benchmark() {
    local shape="$1"
    IFS=',' read -r M N K <<< "$shape"
    local shape_arg="--shape $shape"

    echo ""
    echo "============================================================"
    echo "Benchmark: M=$M, N=$N, K=$K"
    echo "============================================================"
    echo "  Test:   $TEST"
    echo "  Runs:   $RUNS"
    echo "  Kernel: $KERNEL"
    echo "------------------------------------------------------------"

    local runtimes=()
    for i in $(seq 1 "$RUNS"); do
        row=$(rocprofv3 --kernel-trace -- python 7.1_schedule.py --test "$TEST" $shape_arg 2>&1 \
            | grep -oP 'result file: \K[^ ]+' \
            | xargs -I{} sqlite3 {} "SELECT * FROM top WHERE name='$KERNEL';")
        runtime=$(echo "$row" | cut -d'|' -f3)
        if [[ -z "$runtime" ]]; then
            echo "  Run $i: FAILED (no data for kernel '$KERNEL')"
            continue
        fi
        runtimes+=("$runtime")
        printf "  Run %d: %12.3f us\n" "$i" "$runtime"
    done

    if [[ ${#runtimes[@]} -eq 0 ]]; then
        echo "  ERROR: No successful runs for shape $shape."
        return 1
    fi

    python3 -c "
import sys
runtimes = [float(x) for x in sys.argv[1:]]
M, N, K = $M, $N, $K
n = len(runtimes)
avg = sum(runtimes) / n
mn = min(runtimes)
mx = max(runtimes)
flops = 2.0 * M * N * K
tflops_avg = (flops / 1e12) / (avg / 1e6)
tflops_best = (flops / 1e12) / (mn / 1e6)

print('  ----------')
print(f'  Avg runtime:  {avg:,.3f} us')
print(f'  Min runtime:  {mn:,.3f} us')
print(f'  Max runtime:  {mx:,.3f} us')
print(f'  Avg TFLOPS:   {tflops_avg:.4f}')
print(f'  Best TFLOPS:  {tflops_best:.4f}')
" "${runtimes[@]}"
}

export HIP_VISIBLE_DEVICES="$GPU"

echo "============================================================"
echo "Benchmark Suite"
echo "============================================================"
echo "  GPU:    $GPU (HIP_VISIBLE_DEVICES=$GPU)"
echo "  Test:   $TEST"
echo "  Runs:   $RUNS per shape"
echo "  Kernel: $KERNEL"
echo "  Shapes: ${#shapes[@]}"
for s in "${shapes[@]}"; do
    echo "    - $s"
done
echo "============================================================"

for shape in "${shapes[@]}"; do
    run_benchmark "$shape"
done

echo ""
echo "============================================================"
echo "Done."
echo "============================================================"