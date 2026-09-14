"""Fitting-time bandpass filter bank, matching thz-image-explorer's
`src/psf_tool/filters.rs`. (The apply-time filter bank used during actual
deconvolution is decoupled from this and lives in `deconvolution.py`.)"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import bandpass_kaiser

__all__ = ["FrequencySpacing", "NTAPS", "FilterParams", "Filters", "create_filters"]

FrequencySpacing = Literal["log", "linear"]

# Fixed FIR filter length — matches the Rust side, which hardcodes this
# instead of deriving it from the time array length.
NTAPS = 499


@dataclass
class FilterParams:
    n_filters: int = 20
    low_cut: float = 0.1
    high_cut: float = 10.0
    start_freq: float = 0.15
    end_freq: float = 5.0
    win_width: float = 0.5
    frequency_spacing: FrequencySpacing = "log"


@dataclass
class Filters:
    coefficients: np.ndarray  # shape (n_filters, NTAPS)
    center_frequencies: np.ndarray
    fs: float


def create_filters(params: FilterParams, times) -> Filters:
    times = np.asarray(times, dtype=np.float64)
    dt = times[1] - times[0]
    fs = 1.0 / dt

    if params.frequency_spacing == "log":
        center_frequencies = np.geomspace(params.start_freq, params.end_freq, params.n_filters)
    else:
        center_frequencies = np.linspace(params.start_freq, params.end_freq, params.n_filters)

    coefficients = np.zeros((params.n_filters, NTAPS))
    for i in range(params.n_filters):
        if i == 0:
            lowcut = params.low_cut
        else:
            lowcut = np.sqrt(center_frequencies[i - 1] * center_frequencies[i])

        if i == params.n_filters - 1:
            highcut = params.high_cut
        else:
            highcut = np.sqrt(center_frequencies[i] * center_frequencies[i + 1])

        coefficients[i] = bandpass_kaiser(NTAPS, lowcut, highcut, fs, params.win_width)

    return Filters(coefficients=coefficients, center_frequencies=center_frequencies, fs=fs)
