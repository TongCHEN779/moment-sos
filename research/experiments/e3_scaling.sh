#!/bin/bash
# FO for n=10..60 at r=2; MOSEK for n<=30 only (n=40 is OOM-killed: 102091 constraints => ~83 GB Schur complement)
cd "$(dirname "$0")/.."
out=experiments/results/e3.jsonl; : > $out
for n in 10 14 20 26 30; do
  for s in fo MOSEK; do OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 python3 experiments/run_one.py $s $n 2 | grep RESULT | sed 's/^RESULT //' >> $out; done
done
for n in 40 50 60; do OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 python3 experiments/run_one.py fo $n 2 3000 | grep RESULT | sed 's/^RESULT //' >> $out; done
echo finished >> experiments/results/e3.done
