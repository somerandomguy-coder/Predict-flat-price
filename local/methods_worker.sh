#!/bin/bash
# The remaining method runs, strictly one after another at 2 threads. The machine hit OOM with several jobs at once,
# so this must stay a single sequential queue. Run from the repo root with FLATS_DATA_DIR set.
export PYTHONPATH=. METHODS_THREADS=2 OMP_NUM_THREADS=2
PY=/home/nam/Documents/git-repos/active/Predict-flat-price/.venv/bin/python
LOG=out/real/methods
$PY local/methods_s6.py > $LOG/step6.log 2>&1;                  echo "step6 exit $?"
SHARD=1/3 $PY local/methods_s3.py > $LOG/step3_1.log 2>&1;      echo "step3 shard 1 exit $?"
$PY local/methods_final.py f_hl12 f_r800 > $LOG/final.log 2>&1; echo "finalists exit $?"
echo "queue done"
