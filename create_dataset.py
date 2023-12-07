#!/usr/bin/env python3
import json
import h5py
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ecgdl.readers import WFDBReader

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--skip-missing",
    action="store_true",
    help="Skip missing files instead of raising an error",
)
parser.add_argument("--head", type=int, help="Only process the first N files")
parser.add_argument("--annotations", type=str, help="Path to annotations file")
parser.add_argument("--ecg-path", type=str, help="Path to ECG files")
parser.add_argument("--show-errors", action="store_true", help="Show errors")
parser.add_argument("specfile", type=str, help="Path to specfile")
parser.add_argument("record_file", type=str, help="Path to record file")
parser.add_argument("output_file", type=str, help="Path to output file")
args = parser.parse_args()

if args.ecg_path is not None:
    ECG_PATH = Path(args.ecg_path)
else:
    ECG_PATH = Path("/")

if args.annotations is not None:
    annotations = pd.read_csv(args.annotations).set_index("study_id")
else:
    annotations = None

specfile = json.load(open(args.specfile, "r"))

with open(args.record_file, "r") as f:
    all_records = [x.strip() for x in f.readlines()]

record_count = 0
error_count = 0
with h5py.File(args.output_file, "w") as fh:
    for key in specfile:
        fh.attrs[key] = specfile[key]

    if annotations is not None:
        fh.attrs["annotations"] = list(annotations.columns)

    for record in tqdm(all_records):
        record_path = ECG_PATH / record
        if not record_path.with_suffix(".hea").exists():
            if args.skip_missing:
                continue
            else:
                raise RuntimeError(f"Record {record_path} does not exist")

        reader = WFDBReader(
            record_path,
            leads = specfile.get("leads"),
            fs = specfile.get("fs"),
            sig_len = specfile.get("sig_len"),
            remove_baseline=specfile.get("remove_baseline", True),
            remove_powerline=specfile.get("remove_powerline", True),
            require_nonzero_std=specfile.get("require_nonzero_std", True),
            require_no_nan=specfile.get("require_no_nan", True)
        )

        dat, error = reader.read()

        if dat is None:
            error_count += 1
            if args.show_errors:
                print(f"Record {record_path} skipped for reason: {error}")
            continue

        group = fh.create_group(record_path.stem)
        group.create_dataset("ecg", data=dat)

        if annotations is not None:
            record = annotations[annotations.index == int(record_path.stem)].iloc[0]
            for key in record.keys():
                group.attrs[key] = record[key]

        record_count += 1
        if args.head is not None and record_count > args.head:
            break


print(f"Imported {record_count} records with {error_count} errors")