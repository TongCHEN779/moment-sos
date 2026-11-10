#!/bin/bash
# hybrid on all 60 instances; cold ablation on n<=30 and seeds 1-2 at n=40; BM+multiplier certificate (V1) on n=20,30
cd "$(dirname "$0")"
out=../results/bench_hybrid.jsonl; : > $out
run() { OMP_NUM_THREADS=8 python3 hybrid_cert.py "$@" 2>/dev/null | grep RESULT | sed 's/^RESULT //' >> $out; }
sizes() { case "$1" in torus) echo "16 25 36";; *) echo "20 30 40";; esac; }
for fam in g05 pm1s pm1d torus; do for n in $(sizes $fam); do for seed in 1 2 3 4 5; do run $fam $n $seed hybrid; done; done; done
for fam in g05 pm1s pm1d torus; do for n in $(sizes $fam); do
  for seed in 1 2 3 4 5; do
    if [ $n -ge 36 ] && [ $seed -ge 3 ]; then continue; fi
    run $fam $n $seed cold
  done
done; done
for fam in g05 pm1s pm1d torus; do for n in $(sizes $fam); do
  [ $n -ge 36 ] && continue
  for seed in 1 2 3 4 5; do run $fam $n $seed bm; done
done; done
echo done > ../results/bench_hybrid.done
