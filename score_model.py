#!/usr/bin/env python3

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from ecgdl.datasets import HDF5Dataset
import numpy as np
from pathlib import Path
import pandas as pd
import argparse

from ecgdl.models.mayo import MayoModel

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="Only process the first N files")
parser.add_argument("task", type=str, choices=["age", "gender"])
parser.add_argument("target", type=str, choices=["diagnostic_12_lead", "diagnostic_two_lead",
    "monitor_12_lead", "monitor_two_lead"])
parser.add_argument("model", type=str)
args = parser.parse_args()

DATA_PATH = Path("/scratch/ajb5d/ecgdl/data/mimic/")
MODEL_PATH = Path("/scratch/ajb5d/ecgdl/models/mimic/")
TASK = args.task
ARCH = "cnn"
DATA_TARGET = args.target

val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5", 'ecg_age', limit=args.limit)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

CHANNELS = val_dataset[0][0].shape[0]
SAMPLES = val_dataset[0][0].shape[1]
(CHANNELS, SAMPLES)

model = MayoModel(CHANNELS, SAMPLES, 5120, 1).to(device)
model.load_state_dict(torch.load(args.model))
model.eval()

preds = []
labels = []
for X,y in tqdm(iter(val_dataloader), total=len(val_dataloader)):
    with torch.no_grad():
        pred = model(X.to(device)).to('cpu')
        preds.append(pred)
        labels.append(y)
        
preds = torch.cat(preds).squeeze().tolist()
labels = torch.cat(labels).squeeze().tolist()
records = val_dataset._keys

df = pd.DataFrame({'records': [int(x) for x in records], 'y_pred': preds, 'y': labels})
annotations = pd.read_csv("data/mimic/machine_measurements.csv", low_memory=False)
print(f"Writing result to data/scores_{ARCH}_{TASK}_{DATA_TARGET}.csv")
df.set_index('records').join(annotations.set_index('study_id')).to_csv(f"data/scores_{ARCH}_{TASK}_{DATA_TARGET}.csv")