#!/bin/bash
#SBATCH --job-name=zoo_gpu
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=out/logs/zoo/%x_%j.out
cd /project/hnguyen2/mvu9/folder_05_khangcute/Predict-flat-price
export PYTHONPATH=. OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TABPFN_DISABLE_TELEMETRY=1 TMPDIR=${TMPDIR:-/tmp}
nvidia-smi -L
/project/hnguyen2/mvu9/conda_envs/rent_zoo/bin/python local/zoo/member.py "$1" raw w6,w5,w4,w3,w2,w1,B,A,F
