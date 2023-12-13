import wfdb
from pathlib import Path
import numpy as np
from scipy import signal

class WFDBReader:
    def __init__(
        self,
        path: Path,
        leads: [str] = None,
        fs: int = None,
        sig_len: int = None,
        center_and_scale: str = "mean-std",
        remove_baseline: bool = True,
        remove_powerline: bool = True,
        require_nonzero_std: bool = True,
        require_no_nan: bool = True,
        lowpass_filter: float = None,
    ) -> None:

        self.path = path
        self.leads = leads
        self.fs = fs
        self.sig_len = sig_len
        self.center_and_scale = center_and_scale
        self.remove_baseline = remove_baseline
        self.remove_powerline = remove_powerline
        self.require_nonzero_std = require_nonzero_std
        self.require_no_nan = require_no_nan
        self.lowpass_filter = lowpass_filter

    def _filter_remove_baseline(self, fs):
        fc = 0.8  # [Hz], cutoff frequency
        fst = 0.2  # [Hz], rejection band
        rp = 0.5  # [dB], ripple in passband
        rs = 40  # [dB], attenuation in rejection band
        wn = fc / (fs / 2)
        wst = fst / (fs / 2)
        filterorder, aux = signal.ellipord(wn, wst, rp, rs)
        sos = signal.iirfilter(
            filterorder, wn, rp, rs, btype="high", ftype="ellip", output="sos"
        )
        return sos

    def _filter_remove_powerline(self, fs):
        q = 30
        b, a = signal.iirnotch(60.0, q, fs)
        return b, a
    
    def _filter_lowpass(self, cutoff, fs):
        nyq = 0.5 * fs
        order = 5
        normal_cutoff = cutoff / nyq
        return signal.butter(order, normal_cutoff, btype="low", analog=False, output="sos")

    def read(self) -> np.ndarray:
        record = wfdb.rdrecord(self.path.with_suffix(""))

        if self.leads is None:
            self.leads = record.sig_name
            lead_idxs = list(range(len(record.sig_name)))
        else:
            lead_idxs = [record.sig_name.index(x) for x in self.leads]

        current_fs = record.fs

        dat = record.p_signal[:, lead_idxs]

        if self.require_nonzero_std:
            if np.any(dat.std(axis=0) == 0):
                return None, f"Zero std in data (count = {np.sum(dat.std(axis=0) == 0)})"
        
        if self.require_no_nan:
            if np.any(np.isnan(dat)):
                return None, f"NaNs in data (count = {np.sum(np.isnan(dat))}))"

        if self.remove_baseline:
            sos = self._filter_remove_baseline(current_fs)
            dat = signal.sosfilt(sos, dat, axis=0)

        if self.remove_powerline:
            b, a = self._filter_remove_powerline(current_fs)
            dat = signal.filtfilt(b, a, dat, axis=0)

        if self.lowpass_filter is not None:
            sos = self._filter_lowpass(self.lowpass_filter, current_fs)
            dat = signal.sosfilt(sos, dat, axis=0)

        if self.fs is not None and current_fs != self.fs:
            new_len = int(dat.shape[0] * self.fs / current_fs)
            dat = signal.resample(dat, new_len, axis=0)

        if self.sig_len is not None and dat.shape[0] != self.sig_len:
            if dat.shape[0] < self.sig_len:
                dat = np.pad(
                    dat, ((0, self.sig_len - dat.shape[0]), (0, 0)), mode="constant"
                )
            else:
                dat = dat[: self.sig_len, :]

        if self.center_and_scale == "mean_std" :
            dat = (dat - dat.mean(axis=0)) / dat.std(axis=0)
        elif self.center_and_scale == "mean_median_abs":
            dat = (dat - dat.mean(axis=0)) / np.median(np.abs(dat), axis=0)

        return dat, None
