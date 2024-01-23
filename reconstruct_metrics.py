#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re

TARGET = "output/slurm-56984212.out"

parser = argparse.ArgumentParser()
parser.add_argument("outputs", type=str, nargs="+")
args = parser.parse_args()


for output in args.outputs:
    with open(output, "r") as f:
        slurm_output = f.readlines()

    model_path = None
    for line in slurm_output:
        if line.find("Saving model to ") != -1:
            model_path = Path(line.replace("Saving model to ", "").strip())

    if model_path is None:
        print("No model found")
        continue

    if not model_path.exists():
        print(f"Model {model_path} does not exist")
        continue

    print(f"Using model at {model_path}")

    results = []
    for line in slurm_output:
        if line.startswith("Training Results") or line.startswith("Validation Results"):
            target = re.compile(r"(Training|Validation) Results - Epoch\[(\d+)\]: (.*)")
            result = target.match(line.strip())

            dataset = result.group(1).lower()
            epoch = int(result.group(2))
            metrics = json.loads(result.group(3).replace("'", '"'))
            if dataset == "training":
                label = "train"
            elif dataset == "validation":
                label = "val"

            results.append( {
                "epoch": epoch,
                "metrics": metrics,
                "type": label,
            })

    with open(model_path / "progress.json", "w") as f:
        json.dump(results, f)
        print(f"Saved {len(results)} results to {model_path / 'progress.json'}")