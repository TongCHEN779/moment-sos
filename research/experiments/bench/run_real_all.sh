#!/bin/sh
# real Biq Mac rudy g05_60 instances (downloaded from biqmac.aau.at); output: ../results/bench_real.jsonl
for k in 0 1 2 3 4; do python run_real.py /private/tmp/claude-501/-Users-tchen-Documents-Python-moment-sos/4239fa9e-fcce-4def-892a-4e187f81c2aa/scratchpad/biqmac_rudy/g05_60.$k | sed -n 's/^RESULT //p' >> ../results/bench_real.jsonl; done
touch ../results/bench_real.done
