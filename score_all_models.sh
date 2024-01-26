#!/usr/bin/env bash

for MODEL in "$@"; do
    # skip non directories
    if [ ! -d "$MODEL" ]; then
        continue
    fi
    echo "Scoring model: $MODEL"
    # skip directories that have results
    if [ -f "$MODEL/scores_test.csv" ]; then
        echo "Model $MODEL already scored, skipping"
        continue
    fi
    # score model
    sbatch score_model.slurm "$MODEL"
done