#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --output output/slurm-%j.out
#SBATCH --gres=gpu:1 

source .venv/bin/activate
./train_model.py $@
