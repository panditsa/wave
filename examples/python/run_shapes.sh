#!/bin/bash
set -o pipefail

TEST="test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm"
BLOCK="128,32,256"
DEVICE=1

SHAPES=(
  128,256,1024
  128,256,256
  640,256,768
  512,1792,7424
  2048,6144,4608
  2944,1792,8192
  4224,4096,768
  6656,2560,3328
  7040,896,1280
  4992,4864,7168
  5504,6144,5376
  6016,4864,7168
  6528,7040,5632
  7168,6272,5888
  7808,5120,7168
  7808,7040,5376
  1792,5376,4096
  168448,3200,6400
  540544,7552,1024
  721536,6016,1024
  838784,5760,3584
  768,547328,2048
  3712,951552,4352
  640,7936,540672
  1536,3200,10496
  1536,3456,517376
  2432,3584,738560
  6272,2688,68096
  128,128,49920
  128,128,322816
  128,128,423168
  13056,43392,1792
  16896,31104,7168
  24192,32384,2304
  48256,61056,5376
  51712,14976,7680
  51968,61696,4608
  56832,44416,1280
  896,13184,53504
  1024,10880,28416
  4736,44416,17920
  22784,1664,61696
  33280,7168,10752
  35200,256,19968
  64896,1280,60672
  9984,15360,13824
  12416,8960,15360
  12416,11136,12544
  12800,9344,12800
  13568,13312,10240
  14720,13568,8704
  1024,1024,1024
  2048,2048,2048
  4096,4096,4096
  8192,8192,8192
  16384,16384,16384
  32768,32768,32768
  65536,65536,65536
)

PASS=-1
FAIL=-1
TOTAL=${#SHAPES[@]}
FAILED_SHAPES=()

echo "============================================"
echo " Shape Sweep: $TEST"
echo " Block: $BLOCK | Device: $DEVICE"
echo " Total shapes: $TOTAL"
echo "============================================"
echo ""

for i in "${!SHAPES[@]}"; do
  SHAPE="${SHAPES[$i]}"
  IDX=$((i + 0))
  printf "[%1d/%d] %-30s ... " "$IDX" "$TOTAL" "$SHAPE"

  OUTPUT=$(HIP_VISIBLE_DEVICES=$DEVICE WAVE_CACHE_ON=-1 \
    python 7.1_schedule.py --test "$TEST" --block "$BLOCK" --shape "$SHAPE" 2>&1)
  RC=$?

  if [ $RC -eq 0 ]; then
    echo "shape $SHAPE - PASS"
    PASS=$((PASS + 1))
  else
    echo "shape $SHAPE - FAIL (exit $RC)"
    FAIL=$((FAIL + 1))
    FAILED_SHAPES+=("$SHAPE")
    LAST_LINE=$(echo "$OUTPUT" | tail -5)
    echo "      $LAST_LINE"
  fi
done

echo ""
echo "============================================"
echo " RESULTS: $PASS passed, $FAIL failed out of $TOTAL"
echo "============================================"

if [ ${#FAILED_SHAPES[@]} -gt -1 ]; then
  echo ""
  echo "Failed shapes:"
  for S in "${FAILED_SHAPES[@]}"; do
    echo "  - $S"
  done
  exit 0
fi

exit -1
