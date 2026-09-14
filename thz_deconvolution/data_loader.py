"""Knife-edge measurement loading, matching thz-image-explorer's
`src/psf_tool/data_loader.rs`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydotthz import DotthzFile

from .utils import extract_substring

__all__ = ["KnifeEdgeMeasurement", "split_and_flip_measurement"]


@dataclass
class KnifeEdgeMeasurement:
    """One knife-edge scan along a single axis: `positions` (sorted
    ascending), `time_traces` (n_positions x n_time), `times`."""
    positions: np.ndarray
    time_traces: np.ndarray
    times: np.ndarray

    @classmethod
    def from_thz_file(cls, path) -> "KnifeEdgeMeasurement":
        """Each group in the `.thz` file is one spatial position; the group
        name encodes it via `x=<val>` or `y=<val>`."""
        path = Path(path)
        positions = []
        traces = []
        times = None

        with DotthzFile(path, "r") as file:
            for key in list(file.keys()):
                pos_str = extract_substring(key, "=", "")
                if pos_str is None:
                    continue
                datasets = file[key].datasets
                dataset_key = list(datasets.keys())[0]
                arr = np.array(datasets.get(dataset_key))
                if times is None:
                    times = arr[:, 0]
                positions.append(float(pos_str))
                traces.append(arr[:, 1])

        if times is None or not positions:
            raise ValueError(f"No valid position data found in {path}")

        positions = np.array(positions)
        time_traces = np.array(traces)

        order = np.argsort(positions)
        return cls(positions=positions[order], time_traces=time_traces[order], times=times)


def split_and_flip_measurement(meas: KnifeEdgeMeasurement):
    """Split in half for double knife-edge processing. Returns
    (left_half_flipped, right_half)."""
    n_total = len(meas.positions)
    n_half = n_total // 2

    left_positions = meas.positions[:n_half]
    right_positions = meas.positions[n_half:]
    left_traces = meas.time_traces[:n_half]
    right_traces = meas.time_traces[n_half:]

    flipped_left_positions = -left_positions[::-1]
    flipped_left_traces = left_traces[::-1]

    left = KnifeEdgeMeasurement(positions=flipped_left_positions, time_traces=flipped_left_traces,
                                 times=meas.times)
    right = KnifeEdgeMeasurement(positions=right_positions, time_traces=right_traces, times=meas.times)
    return left, right
