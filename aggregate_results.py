#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import pandas as pd
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("models", type=str, nargs="+")
args = parser.parse_args()

models = []
metrics = []

for model in args.models:
    MODEL_DIR = Path(model)
    with open(MODEL_DIR / "spec.json", "r") as f:
        config = json.load(f)

    models.append({
        "model": MODEL_DIR.name,
        "task": config["task"],
        "arch": config["arch"],
        "data": config["data"],
    })

    
    with open(MODEL_DIR / "progress.json", "r") as f:
        results = json.load(f)

    for entry in results:
        for m in entry["metrics"]:
            row = {
                "model": MODEL_DIR.name,
                "epoch": entry["epoch"],
                "set": entry["type"],
                "metric": m,
                "value": entry["metrics"][m],
            }
            metrics.append(deepcopy(row))



df = pd.DataFrame(models)
df.to_csv("models.csv", index=False)

df = pd.DataFrame(metrics)
df.to_csv("metrics.csv", index=False)