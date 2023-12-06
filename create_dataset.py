#!/usr/bin/env python3
import re
import h5py
import wfdb

from pathlib import Path
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import argparse

def ingest_wfdb(p, leads = ["I", "II", "III"], fs = 500):
    record = wfdb.rdrecord(p.with_suffix(""))
    lead_idxs = [record.sig_name.index(x) for x in leads]

    dat = record.p_signal[:, lead_idxs]

    if any(dat.std(axis=0) == 0):
        return None
    
    if record.fs != fs:
        new_len = int(dat.shape[0] * fs / record.fs)
        dat = resample(dat, new_len, axis = 0)

    dat = (dat - dat.mean(axis=0)) / dat.std(axis=0)

    return dat

def lookup_gender(x):
    if all_records[all_records['file_name'] == int(x)].iloc[0].gender == 'M':
        return 1
    return 0

def lookup_age(x):
    return all_records[all_records['file_name'] == int(x)].iloc[0].ecg_age

def create_dataset(input_files, output_file, leads, fs = 500):
    with h5py.File(output_file, "w") as fh:
        fh.attrs["fs"] = fs
        fh.attrs["leads"] = leads
        for file in tqdm(input_files):
            dat = ingest_wfdb(file, leads, fs)
            if dat is None:
                continue
            group = fh.create_group(file.stem)
            group.create_dataset("ecg", data=dat)
            group.create_dataset("gender", data=lookup_gender(file.stem))
            group.create_dataset("age", data=lookup_age(file.stem))

parser = argparse.ArgumentParser()
parser.add_argument('--ecg_path', type=str, required=True)
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--head', type=int)
parser.add_argument('--fs', type=int, default=500)
parser.add_argument('--leads', type=str, nargs="+", required=True)
args = parser.parse_args()

ECG_PATH = Path(args.ecg_path)

all_records = pd.read_csv(args.file)
all_files = [ECG_PATH / Path(x) for x in all_records['path']]

if not args.head is None:
    all_files = all_files[:args.head]

create_dataset(all_files, args.output, args.leads, fs=args.fs)