#!/bin/bash
#SBATCH --job-name=zoo_write
#SBATCH --partition=batch
#SBATCH --exclude=compute-0-8,compute-0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=out/logs/%x_%j.out
cd /project/hnguyen2/mvu9/folder_05_khangcute/Predict-flat-price
export PYTHONPATH=. OMP_NUM_THREADS=4
/project/hnguyen2/mvu9/conda_envs/rent_zoo/bin/python local/zoo/stack.py "$@"
