#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from ecgdl.datasets import HDF5Dataset
from ecgdl.models.mayo import MayoModel
from ecgdl.models.resnet import ResNet1d
from ecgdl.models.mha import MultiHeadAttention
from ecgdl.models.resnetmha import ResNetMHA

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
parser.add_argument("model", type=str)
args = parser.parse_args()

DATA_PATH = Path("/scratch/ajb5d/ecgdl/data/mimic/")
MODEL_PATH = Path("/scratch/ajb5d/ecgdl/models/mimic/")

MODEL_DIR = MODEL_PATH / args.model

if not MODEL_DIR.exists():
    raise ValueError(f"Model {args.model} does not exist")

with open(MODEL_DIR / "spec.json", "r") as f:
    config = json.load(f)

TASK = config['task']
ARCH = config['arch']
DATA_TARGET = config['data']

assert(TASK in ['gender','age'])
assert(ARCH in ['resnet','cnn', 'mha', 'resnetmha'])

if TASK == "gender":
    test_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_test.h5", 
        'gender',
        'M',
        limit=args.limit)
    val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5",
        'gender',
        'M',
        limit=args.limit)

if TASK == "age":
    test_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_test.h5",
        'ecg_age',
        limit=args.limit,
        filter_func=lambda x: x < 90)
    val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5",
        'ecg_age',
        limit=args.limit,
        filter_func=lambda x: x < 90)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

CHANNELS = test_dataset[0][0].shape[0]
SAMPLES = test_dataset[0][0].shape[1]
(CHANNELS, SAMPLES)

if ARCH == "resnet":
    model = ResNet1d(
        input_dim = (CHANNELS, SAMPLES),
        blocks_dim = list(zip(config["model"]["net_filter_sizes"], config["model"]["net_sequence_lengths"])),
        n_classes = 1,
        kernel_size = 17,
        dropout_rate = 0.8).to(device)
elif ARCH == "cnn":
    model = MayoModel(CHANNELS, SAMPLES, 5120, 1).to(device)
elif ARCH == "mha":
    model = MultiHeadAttention(1).to(device)
elif ARCH == "resnetmha":
        model = ResNetMHA(
            input_dim = (CHANNELS, SAMPLES),
            blocks_dim = list(zip(config["model"]["net_filter_sizes"], config["model"]["net_sequence_lengths"])),
            n_classes = 1,
            kernel_size = 17,
            dropout_rate = 0.3,
            n_heads = 4).to(device)

all_checkpoints = MODEL_DIR.glob("checkpoint_*.pt")
all_filenames = [x.with_suffix("").name for x in all_checkpoints]
all_scores = [float(x.replace("checkpoint_model_", "")) for x in all_filenames]
best_score_idx = np.argmax(all_scores)

best_checkpoint = MODEL_DIR / f"{all_filenames[best_score_idx]}.pt"
print(f"Loading best checkpoint {best_checkpoint}")
model.load_state_dict(torch.load(best_checkpoint, map_location=device))
model.eval()

preds = []
labels = []
for X,y in tqdm(iter(test_dataloader), total=len(test_dataloader)):
    with torch.no_grad():
        pred = model(X.to(device)).to('cpu')
        preds.append(pred)
        labels.append(y)
        
preds = torch.cat(preds).squeeze().tolist()
labels = torch.cat(labels).squeeze().tolist()
records = test_dataset._keys

df = pd.DataFrame({'records': [int(x) for x in records], 'y_pred': preds, 'y': labels})
print(f"Writing result to {MODEL_DIR / 'scores_test.csv'} ")
(
    df
        .set_index('records')
        .to_csv(MODEL_DIR / "scores_test.csv")
)

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
print(f"Writing result to {MODEL_DIR / 'scores_val.csv'} ")
(
    df
        .set_index('records')
        .to_csv(MODEL_DIR / "scores_val.csv")
)