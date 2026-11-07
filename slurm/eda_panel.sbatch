#!/bin/bash
#SBATCH --job-name=rent_eda_panel
#SBATCH --partition=batch
#SBATCH --exclude=compute-0-8,compute-0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=out/logs/%x_%j.out
cd /project/hnguyen2/mvu9/folder_05_khangcute/Predict-flat-price
export PYTHONPATH=. OMP_NUM_THREADS=8
/project/hnguyen2/mvu9/conda_envs/rent_zoo/bin/python local/eda_panel.py
