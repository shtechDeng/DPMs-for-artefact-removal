#!/bin/bash
#SBATCH -J diffusion_initial
#SBATCH -p bme_gpu4
#SBATCH --gres=gpu:1
#SBATCH --exclude bme_gpu[01-10]
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH --output=slurm_out/diffusion_initial.out
#SBATCH --error=slurm_out/diffusion_initial.err

hostname
date
nvidia-smi

python main.py --config bcp.yml --exp exp --doc diffusion_initial --ni

date