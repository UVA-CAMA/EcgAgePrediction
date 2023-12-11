#!/usr/bin/env python3
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from ecgdl.datasets import HDF5Dataset
import numpy as np
import torchinfo

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from ignite.handlers import ModelCheckpoint, TerminateOnNan, ReduceLROnPlateauScheduler
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.metrics import ROC_AUC

from pathlib import Path
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
args = parser.parse_args()

DATA_PATH = Path("/scratch/ajb5d/ecgdl/data/mimic/")
MODEL_PATH = Path("/scratch/ajb5d/ecgdl/models/mimic/")
TASK = args.task
ARCH = "cnn"
DATA_TARGET = args.target

if args.task == "gender":
    train_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_train.h5", 'gender', 'M',  limit=args.limit)
    val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5", 'gender', 'M', limit=args.limit)

if args.task == "age":
    train_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_train.h5",
        'ecg_age',
        limit=args.limit,
        filter_func=lambda x: x < 90)
    val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5",
        'ecg_age',
        limit=args.limit,
        filter_func=lambda x: x < 90)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

CHANNELS = train_dataset[0][0].shape[0]
SAMPLES = train_dataset[0][0].shape[1]
(CHANNELS, SAMPLES)

model = MayoModel(CHANNELS, SAMPLES, 5120, 1).to(device)
torchinfo.summary(model, input_size=(128, CHANNELS, SAMPLES))

learning_rate = 1e-3
batch_size = 64
epochs = 5

def output_transform_logit(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y

if args.task == "gender":
    criterion = nn.BCEWithLogitsLoss()
    val_metrics = {
        "loss": Loss(criterion),    
        "auc": ROC_AUC(output_transform_logit),
    }

if args.task == "age":
    criterion = nn.MSELoss()
    val_metrics = {
        "loss": Loss(criterion),    
        "mae": MeanAbsoluteError()
    }

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

save_path = MODEL_PATH / TASK / f"{ARCH}_{DATA_TARGET}"
print(f"Saving model to {save_path}")
if not save_path.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)

trainer = create_supervised_trainer(model, optimizer, criterion, device)

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

log_interval = 500
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
    
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_dataloader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}]")
    print(metrics)

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_dataloader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}]")
    print(metrics)
    
model_checkpoint = ModelCheckpoint(save_path, 'checkpoint', n_saved=2, create_dir=True, require_empty=False)
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

scheduler = ReduceLROnPlateauScheduler(optimizer, metric_name="loss", patience=5, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, scheduler)

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
state = trainer.run(train_dataloader, max_epochs=50)

print(state.metrics)

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, 
    save_path / "model.pth")