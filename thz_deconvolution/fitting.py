"""Per-filter beam-width/beam-center fitting, matching thz-image-explorer's
`src/psf_tool/fitting.rs`. Unlike the old (pre-2.0) pipeline, X and Y are
fitted independently — the caller runs this once per axis file."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.special
from scipy import signal
from scipy.optimize import curve_fit

__all__ = [
    "error_function", "compute_intensity",
    "MeanBeamFit", "fit_mean_beam",
    "BeamFitParams", "BeamWidthFits", "fit_beam_widths",
]


def error_function(x, x0, w):
    return (1.0 + scipy.special.erf(math.sqrt(2) * (np.asarray(x) - x0) / w)) / 2.0


def compute_intensity(traces: np.ndarray) -> np.ndarray:
    """Sum-of-squares intensity per position, normalized to [0, 1]."""
    intensity = np.sum(traces ** 2, axis=1)
    min_val = np.min(intensity)
    max_val = np.max(intensity)
    if abs(max_val - min_val) > 1e-10:
        intensity = (intensity - min_val) / (max_val - min_val)
    return intensity


@dataclass
class MeanBeamFit:
    x0: float
    popt: np.ndarray  # [x0, w]


def fit_mean_beam(positions, traces) -> MeanBeamFit:
    """Fit an error function to the (unfiltered) intensity profile to get an
    initial center/width estimate."""
    intensity = compute_intensity(traces)
    popt, _ = curve_fit(error_function, positions, intensity, p0=[0.0, 10.0], maxfev=8000)
    return MeanBeamFit(x0=popt[0], popt=popt)


@dataclass
class BeamFitParams:
    w_max: float = 30.0
    use_monotonicity_constraint: bool = True


@dataclass
class BeamWidthFits:
    popt: np.ndarray  # shape (n_filters, 2): [center, width] per filter


def fit_beam_widths(mean_fit: MeanBeamFit, positions, traces, filters: np.ndarray,
                     fit_params: BeamFitParams,
                     progress_callback: Optional[Callable[[int, int], bool]] = None) -> BeamWidthFits:
    """Fit beam center/width per filter frequency, carrying the previous
    filter's fit forward as the initial guess (and, if
    `use_monotonicity_constraint`, tightening the bounds around it)."""
    positions = np.asarray(positions, dtype=np.float64)
    traces = np.asarray(traces, dtype=np.float64)
    n_filters = filters.shape[0]
    popt_out = np.zeros((n_filters, 2))

    w_max = fit_params.w_max
    range_max = w_max * 1.5
    bounds = ([-range_max / 2.0, 0.01], [range_max / 2.0, w_max])
    p0 = [mean_fit.popt[0], w_max]

    for nf in range(n_filters):
        filtered = np.array([signal.convolve(traces[i], filters[nf], mode='same')
                              for i in range(traces.shape[0])])
        intensity = compute_intensity(filtered)

        popt, _ = curve_fit(error_function, positions, intensity, p0=p0, bounds=bounds, maxfev=8000)

        offset, w = popt[0], popt[1]
        if fit_params.use_monotonicity_constraint:
            bounds = ([-w / 2.0 + offset, 0.0], [w / 2.0 + offset, w])
        else:
            bounds = ([-range_max / 2.0, 0.01], [range_max / 2.0, w_max])

        popt_out[nf, 0] = popt[0]
        popt_out[nf, 1] = abs(popt[1])
        p0 = popt

        if progress_callback is not None and not progress_callback(nf + 1, n_filters):
            raise RuntimeError("Cancelled")

    return BeamWidthFits(popt=popt_out)
