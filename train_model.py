#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchinfo

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanAbsoluteError, RunningAverage
from ignite.handlers import ModelCheckpoint, Checkpoint, TerminateOnNan, ReduceLROnPlateauScheduler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.metrics import ROC_AUC


from ecgdl.datasets import HDF5Dataset
from ecgdl.models.mayo import MayoModel
from ecgdl.models.resnet import ResNet1d
from ecgdl.models.mha import MultiHeadAttention

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
parser.add_argument("--verbose", action="store_true")
parser.add_argument("specfile", type=str)
args = parser.parse_args()

with open(args.specfile, "r") as f:
    specfile = json.load(f)

DATA_PATH = Path("/scratch/ajb5d/ecgdl/data/mimic/")
MODEL_PATH = Path("/scratch/ajb5d/ecgdl/models/mimic/")
TASK = specfile.get("task")
ARCH = specfile.get("arch")
DATA_TARGET = specfile.get("data")

assert(TASK in ['gender','age'])
assert(ARCH in ['resnet','cnn', 'mha'])

if TASK == "gender":
    train_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_train.h5", 
        'gender',
        'M',
        limit=args.limit)
    val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5",
        'gender',
        'M',
        limit=args.limit)

if TASK == "age":
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

if ARCH == "resnet":
    model = ResNet1d(
        input_dim = (CHANNELS, SAMPLES),
        blocks_dim = list(zip(specfile["model"]["net_filter_sizes"], specfile["model"]["net_sequence_lengths"])),
        n_classes = 1,
        kernel_size = 17,
        dropout_rate = 0.8).to(device)
elif ARCH == "cnn":
    model = MayoModel(CHANNELS, SAMPLES, 5120, 1).to(device)
elif ARCH == "mha":
    model = MultiHeadAttention(1).to(device)

torchinfo.summary(model, input_size=(128, CHANNELS, SAMPLES))

learning_rate = specfile.get("lr", 1e-3)
epochs = specfile.get("epochs", 50)

def output_transform_logit(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y

if TASK == "gender":
    criterion = nn.BCEWithLogitsLoss()
    val_metrics = {
        "loss": Loss(criterion),    
        "auc": ROC_AUC(output_transform_logit),

    }

if TASK == "age":
    criterion = nn.MSELoss()
    val_metrics = {
        "loss": Loss(criterion),    
        "mae": MeanAbsoluteError()
    }

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_label = uuid.uuid1()
save_path = MODEL_PATH / f"{model_label}/"
print(f"Saving model to {save_path}")
if not save_path.exists():
    save_path.mkdir()

with open(save_path / "spec.json", "w") as f:
    json.dump(specfile, f)

with open(save_path / "settings.json", "w") as f:
    d = {
        "lr": learning_rate,
        "epochs": epochs,
        "limit": args.limit,
    }

    json.dump(d, f)


trainer = create_supervised_trainer(model, optimizer, criterion, device)
train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_dataloader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}]: {metrics}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_dataloader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}]: {metrics}")

neg_loss_score = Checkpoint.get_default_score_fn("loss", -1.0)
model_checkpoint = ModelCheckpoint(save_path, 'checkpoint', n_saved=2, create_dir=True, require_empty=False, score_function=neg_loss_score)
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

scheduler = ReduceLROnPlateauScheduler(optimizer, metric_name="loss", patience=5, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, scheduler)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

if args.verbose:
    ProgressBar().attach(trainer, ['loss'])
    ProgressBar(desc="Train Evalaution").attach(train_evaluator)
    ProgressBar(desc="Validation Evalaution").attach(val_evaluator)

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
state = trainer.run(train_dataloader, max_epochs=epochs)

print(state.metrics)

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, 
    save_path / "final_model.pth")