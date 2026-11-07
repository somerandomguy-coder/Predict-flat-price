#!/bin/bash
#SBATCH --job-name=zoo_cpu
#SBATCH --partition=batch
#SBATCH --exclude=compute-0-8,compute-0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:40:00
#SBATCH --output=out/logs/zoo/%x_%A_%a.out
cd /project/hnguyen2/mvu9/folder_05_khangcute/Predict-flat-price
export PYTHONPATH=. OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
/project/hnguyen2/mvu9/conda_envs/rent_zoo/bin/python local/zoo/member.py $LINE
