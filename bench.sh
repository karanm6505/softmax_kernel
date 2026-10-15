#!/usr/bin/env bash
# Sweep row widths while keeping the total work constant (2^25 floats = 128 MB),
# so every shape moves the same number of bytes and timings are comparable.
set -euo pipefail

OUT=${1:-results.csv}
TOTAL=$((1 << 25))

echo "kernel,rows,cols,median_ms,eff_GBps,max_abs_err,check" > "$OUT"
for COLS in 128 512 1024 4096 16384 32768; do
    ROWS=$((TOTAL / COLS))
    ./softmax "$ROWS" "$COLS" 50 >> "$OUT"
    python3 bench_torch.py "$ROWS" "$COLS" >> "$OUT"
done
cat "$OUT"
