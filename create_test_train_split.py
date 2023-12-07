#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DATA_PATH = Path("./data")

patients = pd.read_csv(BASE_DATA_PATH / "mimic" / "patients.csv")
patients.set_index('subject_id', inplace=True)

ecg_data = pd.read_csv(BASE_DATA_PATH / "mimic" / "record_list.csv")
ecg_data['ecg_time'] = pd.to_datetime(ecg_data['ecg_time'])
ecg_data = ecg_data.join(patients, on='subject_id', how='left')
ecg_data['ecg_age'] = ecg_data['ecg_time'].dt.year - ecg_data['anchor_year'] + ecg_data['anchor_age']

all_patients = ecg_data['subject_id'].unique()

trainval_patients, test_patients = train_test_split(all_patients, test_size=0.2, random_state=42)
train_patients, val_patients = train_test_split(trainval_patients, test_size=0.2, random_state=42)

sets = {
    'train': train_patients,
    'val': val_patients,
    'test': test_patients
}

ecg_data[['study_id', 'gender', 'ecg_age']].to_csv(
    BASE_DATA_PATH / "mimic" / "derived_ecg_annotations.csv", index=False)

for s in sets:
    with open(BASE_DATA_PATH / "mimic" / f"RECORDS_{s}.txt", 'w') as f:
        for x in ecg_data[ecg_data['subject_id'].isin(sets[s])]['path']:
            f.write(f"{x}\n")