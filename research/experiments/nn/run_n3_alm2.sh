#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
python3 -u n3_alm.py 5 40,40 11 0.1 1500 100 600 > ../results/nn_alm2_A.log 2>&1
python3 -u n3_alm.py 6 48,48 12 0.1 1500 100 600 > ../results/nn_alm2_C.log 2>&1
echo done > ../results/nn_alm2.done
