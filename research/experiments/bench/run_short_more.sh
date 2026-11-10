#!/bin/bash
# remaining short-schedule runs: n=36/40 seeds 3-5 of the four families, then 3-regular graphs n=50 (seeds 1-5) and n=60 (seeds 1-3)
cd "$(dirname "$0")"
out=../results/bench_hybrid.jsonl
run() { OMP_NUM_THREADS=8 python3 hybrid_cert.py "$@" 2>/dev/null | grep RESULT | sed 's/^RESULT //' >> $out; }
for fam in g05 pm1s pm1d; do for seed in 3 4 5; do run $fam 40 $seed cold; done; done
for seed in 3 4 5; do run torus 36 $seed cold; done
echo small-done > ../results/bench_short_more.stage
for seed in 1 2 3 4 5; do run reg3 50 $seed cold; done
for seed in 1 2 3; do run reg3 60 $seed cold; done
echo done > ../results/bench_short_more.done
