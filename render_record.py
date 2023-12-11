#!/usr/bin/env python3

from ecgdl.datasets import HDF5Dataset
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, choices=["age", "gender"])
parser.add_argument("target", type=str, choices=["diagnostic_12_lead", "diagnostic_two_lead",
    "monitor_12_lead", "monitor_two_lead"])
parser.add_argument("input", type=str)
args = parser.parse_args()

DATA_PATH = Path("/scratch/ajb5d/ecgdl/data/mimic/")
MODEL_PATH = Path("/scratch/ajb5d/ecgdl/models/mimic/")
TASK = args.task
ARCH = "cnn"
DATA_TARGET = args.target

val_dataset = HDF5Dataset(DATA_PATH / f"{DATA_TARGET}_val.h5", None)

dat = val_dataset.lookup(args.input)["ecg"][()]
leads = val_dataset._hdf_handle.attrs["leads"]

ecg_dat = pd.read_csv("data/mimic/machine_measurements.csv", low_memory=False).set_index('study_id')
rec = ecg_dat[ecg_dat.index == int(args.input)].iloc[0]

p_duration = rec.p_end - rec.p_onset
pr_interval = rec.qrs_onset - rec.p_onset
qrs_duration = rec.qrs_end - rec.qrs_onset
qt_interval = rec.t_end - rec.qrs_onset
qrs_axis = rec.qrs_axis
p_axis = rec.p_axis
rr_interval = rec.rr_interval

print(dat.shape)
fix, ax = plt.subplots(dat.shape[1], 1, figsize=(20,10))

for i in range(dat.shape[1]):
    ax[i].plot(dat[:,i])
    ax[i].set_ylabel(leads[i])

ax[0].set_title(f"{args.input} Dur: p {p_duration} qrs {qrs_duration} Int: qt {qt_interval} pr {pr_interval} rr {rr_interval}")
plt.savefig(f"{args.input}.png")