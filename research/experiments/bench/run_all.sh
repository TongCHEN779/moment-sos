#!/bin/bash
# Order-2 MAX-CUT benchmark: 4 families x 3 sizes x 5 seeds; MOSEK reference where n <= 25.
cd "$(dirname "$0")"
out=../results/bench.jsonl; : > $out
run() { python3 run_bench.py "$@" 2>/dev/null | grep RESULT | sed 's/^RESULT //' >> $out; }
for size in small mid large; do
  for fam in g05 pm1s pm1d torus; do
    case "$fam-$size" in
      torus-small) n=16;; torus-mid) n=25;; torus-large) n=36;;
      *-small) n=20;; *-mid) n=30;; *-large) n=40;;
    esac
    mk=0; [ $n -le 25 ] && mk=1
    for seed in 1 2 3 4 5; do run $fam $n $seed $mk; done
  done
done
echo done > ../results/bench.done
