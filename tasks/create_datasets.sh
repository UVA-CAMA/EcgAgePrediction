#!/usr/bin/env bash

COHORT_NAME=$(basename -s .json "$1")

echo "Building for $COHORT_NAME"

sbatch create_dataset.slurm \
    --annotations data/mimic/derived_ecg_annotations.csv \
    --show-errors \
    --ecg-path /scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    "$1" \
    data/mimic/RECORDS_train.txt \
    /scratch/ajb5d/ecgdl/data/mimic/"$COHORT_NAME"_train.h5

sbatch create_dataset.slurm \
    --annotations data/mimic/derived_ecg_annotations.csv \
    --show-errors \
    --ecg-path /scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    "$1" \
    data/mimic/RECORDS_test.txt \
    /scratch/ajb5d/ecgdl/data/mimic/"$COHORT_NAME"_test.h5

sbatch create_dataset.slurm \
    --annotations data/mimic/derived_ecg_annotations.csv \
    --show-errors \
    --ecg-path /scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    "$1" \
    data/mimic/RECORDS_val.txt \
    /scratch/ajb5d/ecgdl/data/mimic/"$COHORT_NAME"_val.h5



