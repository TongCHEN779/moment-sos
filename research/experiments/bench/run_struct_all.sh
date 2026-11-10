#!/bin/bash
cd "$(dirname "$0")"
out=../results/bench_struct.jsonl; : > $out
run() { python3 run_struct.py "$@" 2>/dev/null | grep RESULT | sed 's/^RESULT //' >> $out; }
for n in 8 10 12; do for seed in 1 2 3 4 5; do run boxqp $n $seed; done; done
for n in 12 16 20 24; do for seed in 1 2 3 4 5; do run chain_sparse $n $seed; done; done
for n in 12 16; do for seed in 1 2 3 4 5; do run chain_dense $n $seed; done; done
echo done > ../results/bench_struct.done
