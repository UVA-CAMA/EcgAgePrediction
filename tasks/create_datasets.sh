#!/usr/bin/env bash

sbatch create_dataset.slurm \
    --ecg_path /scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --file data/mimic/test_ecgs.csv \
    --output data/mimic/test_ecgs_II.h5 \
    --leads II \
    --fs 250

sbatch create_dataset.slurm \
    --ecg_path /scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --file data/mimic/train_ecgs.csv \
    --output data/mimic/train_ecgs_II.h5 \
    --leads II \
    --fs 250

sbatch create_dataset.slurm \
    --ecg_path /scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --file data/mimic/val_ecgs.csv \
    --output data/mimic/val_ecgs_II.h5 \
    --leads II \
    --fs 250

    