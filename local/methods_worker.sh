#!/bin/bash
# One single-thread worker of three. Under CPU contention one thread per process beat three threads in one process.
# usage: local/methods_worker.sh <0|1|2>, from the repo root, with FLATS_DATA_DIR set.
export PYTHONPATH=. METHODS_THREADS=1 OMP_NUM_THREADS=1 SHARD="$1/3"
PY=/home/nam/Documents/git-repos/active/Predict-flat-price/.venv/bin/python
LOG=out/real/methods
case "$1" in
  0) $PY local/methods_s7.py > $LOG/step7.log 2>&1
     $PY local/methods_s3.py > $LOG/step3_0.log 2>&1
     $PY local/methods_s6.py > $LOG/step6.log 2>&1 ;;
  1) $PY local/methods_s4.py > $LOG/step4.log 2>&1
     $PY local/methods_s3.py > $LOG/step3_1.log 2>&1 ;;
  2) $PY local/methods_s3.py > $LOG/step3_2.log 2>&1
     $PY local/methods_s5.py > $LOG/step5.log 2>&1 ;;
esac
echo "worker $1 done"
