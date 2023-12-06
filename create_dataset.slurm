#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --output output/slurm-%j.out

source env/bin/activate
./create_dataset.py $@