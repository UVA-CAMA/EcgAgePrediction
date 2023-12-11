import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path:str, output_key:str=None, reference_level:str=None, limit:int=None, filter_func=None):
        """
        Create a dataset from an HDF5 file.

        Parameters
        ----------
        hdf5_path : str
            Path to the HDF5 file.

        output_key : str, optional
            Key to use for the output label. If None, no label will be returned.

        reference_key : str, optional
            If specified, the output label will be 1.0 if the value of output_key
            matches this value, and 0.0 otherwise. If None, the value of output_key
            will be returned as-is.

        limit : int, optional
            if specified, use only the first limit data points (for testing)
        """

        self.hdf5_path = hdf5_path
        self.output_key = output_key
        self.reference_level = reference_level
        self.filter_func = filter_func

        self._hdf_handle = h5py.File(self.hdf5_path, "r")
        self._keys = []

        for key in tqdm(self._hdf_handle.keys()):
            if self.reference_level is not None:
                self._keys.append(key)
            else:
                val = self._hdf_handle[key].attrs[self.output_key]
                if not np.isnan(val):
                    ## No filter or filter is true
                    if not self.filter_func or self.filter_func(val):
                        self._keys.append(key)
            if limit:
                if len(self._keys) >= limit:
                    break

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        record = self._hdf_handle[self._keys[idx]]

        dat = record["ecg"][()]
        if self.output_key is not None:
            if self.reference_level is not None:
                label = (
                    1.0 if record.attrs[self.output_key] == self.reference_level else 0.0
                )
            else:
                label = record.attrs[self.output_key]
        else:
            label = None

        return torch.tensor(dat, dtype=torch.float32).T, torch.tensor([label], dtype=torch.float32)
