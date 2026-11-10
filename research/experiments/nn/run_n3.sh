#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
python3 -u n3_scale.py 5 40,40 11 0.1 2400 1 1 > ../results/nn_n3_A.log 2>&1
python3 -u n3_scale.py 5 40,40 11 0.2 2400 0 0 > ../results/nn_n3_B.log 2>&1
python3 -u n3_scale.py 6 48,48 12 0.1 2400 0 0 > ../results/nn_n3_C.log 2>&1
echo done > ../results/nn_n3.done
