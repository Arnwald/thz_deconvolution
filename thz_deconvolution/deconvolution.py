import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal
from tqdm import tqdm

from .data_loader import KnifeEdgeMeasurement, split_and_flip_measurement
from .filters import FilterParams, NTAPS, create_filters
from .fitting import BeamFitParams, fit_beam_widths, fit_mean_beam
from .psf import PSF, create_psf_2d, gaussian
from .utils import bandpass_kaiser, range_max_min, richardson_lucy_unclipped

__all__ = ["Deconvolution"]


def _build_runtime_filter_bank(times, n_filters, start_freq, end_freq, win_width):
    """Filter bank rebuilt fresh at apply-time from the scan's own time axis,
    decoupled from whatever bank was used to fit the PSF. First filter is a
    pure lowpass, last is a pure highpass to Nyquist — matches
    thz-image-explorer's `src/filters/deconvolution.rs::create_filter_bank`.
    """
    times = np.asarray(times, dtype=np.float64)
    dt = times[1] - times[0]
    fs = 1.0 / dt

    center_frequencies = np.geomspace(start_freq, end_freq, n_filters)

    coefficients = np.zeros((n_filters, NTAPS))
    for i in range(n_filters):
        lowcut = 0.0 if i == 0 else np.sqrt(center_frequencies[i - 1] * center_frequencies[i])
        highcut = (0.5 * fs if i == n_filters - 1
                   else np.sqrt(center_frequencies[i] * center_frequencies[i + 1]))
        coefficients[i] = bandpass_kaiser(NTAPS, lowcut, highcut, fs, win_width)

    return coefficients, center_frequencies, fs


class Deconvolution:

    def __init__(self, knife_edge_x_path: Optional[Path] = None, knife_edge_y_path: Optional[Path] = None,
                 psf_path: Optional[Path] = None,
                 low_cut=0.1,
                 high_cut=10.0,
                 start_freq=0.15,
                 end_freq=5.0,
                 win_width=0.5,
                 n_filters=20,
                 frequency_spacing='log',
                 w_max=30.0,
                 use_monotonicity_constraint=True,
                 npz_path_to_save: Optional[Path] = None):

        # Keep the filter bank valid when a caller raises low_cut without also
        # raising start_freq (filter 0 spans [low_cut, geomean(f0, f1)]).
        start_freq = max(start_freq, low_cut)

        self._config = {
            'low_cut': low_cut,
            'high_cut': high_cut,
            'start_freq': start_freq,
            'end_freq': end_freq,
            'win_width': win_width,
            'n_filters': n_filters,
            'frequency_spacing': frequency_spacing,
            'w_max': w_max,
            'use_monotonicity_constraint': use_monotonicity_constraint,
            'knife_edge_x_path': str(knife_edge_x_path) if knife_edge_x_path else None,
            'knife_edge_y_path': str(knife_edge_y_path) if knife_edge_y_path else None,
            'psf_path': str(psf_path) if psf_path else None,
        }

        # `_fitting_times` is the time axis used to build the fitting-time
        # filter bank; it's reused as the default apply-time filter bank
        # basis when the caller doesn't pass one explicitly (only available
        # when the PSF was fitted here rather than loaded from psf_path).
        self._fitting_times = None

        if psf_path is not None:
            self.psf = PSF.load_npz(psf_path)
            return

        if knife_edge_x_path is None and knife_edge_y_path is None:
            raise ValueError("At least a psf or one of the knife edge measurement files must be provided.")

        filter_params = FilterParams(n_filters=n_filters, low_cut=low_cut, high_cut=high_cut,
                                      start_freq=start_freq, end_freq=end_freq, win_width=win_width,
                                      frequency_spacing=frequency_spacing)
        fit_params = BeamFitParams(w_max=w_max, use_monotonicity_constraint=use_monotonicity_constraint)

        x_meas = KnifeEdgeMeasurement.from_thz_file(knife_edge_x_path) if knife_edge_x_path else None
        y_meas = KnifeEdgeMeasurement.from_thz_file(knife_edge_y_path) if knife_edge_y_path else None

        self._fitting_times = x_meas.times if x_meas is not None else y_meas.times
        filters = create_filters(filter_params, self._fitting_times)

        def fit_axis(meas: KnifeEdgeMeasurement) -> np.ndarray:
            left, right = split_and_flip_measurement(meas)
            mean_left = fit_mean_beam(left.positions, left.time_traces)
            fits_left = fit_beam_widths(mean_left, left.positions, left.time_traces,
                                         filters.coefficients, fit_params)
            mean_right = fit_mean_beam(right.positions, right.time_traces)
            fits_right = fit_beam_widths(mean_right, right.positions, right.time_traces,
                                          filters.coefficients, fit_params)

            # Average left/right, undoing the left-half sign flip, then recenter.
            popt_avg = np.empty_like(fits_left.popt)
            popt_avg[:, 0] = (-fits_left.popt[:, 0] + fits_right.popt[:, 0]) / 2.0
            popt_avg[:, 1] = (fits_left.popt[:, 1] + fits_right.popt[:, 1]) / 2.0
            mean_pos = np.mean(popt_avg[:, 0])
            popt_avg[:, 0] -= mean_pos
            return popt_avg

        popt_x = fit_axis(x_meas) if x_meas is not None else None
        popt_y = fit_axis(y_meas) if y_meas is not None else None

        freqs = filters.center_frequencies
        if popt_x is not None and popt_y is not None:
            wx, x0 = np.abs(popt_x[:, 1]), popt_x[:, 0]
            wy, y0 = np.abs(popt_y[:, 1]), popt_y[:, 0]
        elif popt_x is not None:
            wx, x0 = np.abs(popt_x[:, 1]), popt_x[:, 0]
            wy, y0 = wx, x0
        else:
            wy, y0 = np.abs(popt_y[:, 1]), popt_y[:, 0]
            wx, x0 = wy, y0

        self.psf = PSF.fit_from_data(freqs, wx, wy, x0, y0)

        default_source = Path(knife_edge_x_path) if knife_edge_x_path is not None else Path(knife_edge_y_path)
        save_path = (Path(npz_path_to_save) if npz_path_to_save is not None
                     else default_source.parent.with_name("psf_new").with_suffix(".npz"))
        self.psf.save_npz(save_path)

    def _create_config_hash(self, scan_params):
        """Create a hash from configuration and scan parameters."""
        combined_config = {**self._config, **scan_params}
        config_str = str(sorted(combined_config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, config_hash):
        """Get the cache file path for a given configuration hash."""
        temp_dir = Path(tempfile.gettempdir()) / "thz_deconvolution_cache"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir / f"deconv_{config_hash}.pkl"

    def _save_cached_result(self, config_hash, deconvolved_traces):
        """Save deconvolved traces to cache."""
        cache_path = self._get_cache_path(config_hash)
        with open(cache_path, 'wb') as f:
            pickle.dump(deconvolved_traces, f)

    def _load_cached_result(self, config_hash):
        """Load deconvolved traces from cache if available."""
        cache_path = self._get_cache_path(config_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, IOError):
                cache_path.unlink(missing_ok=True)
        return None

    def clear_cache(self):
        """Clear all cached deconvolution results."""
        temp_dir = Path(tempfile.gettempdir()) / "thz_deconvolution_cache"
        if temp_dir.exists():
            for cache_file in temp_dir.glob("deconv_*.pkl"):
                cache_file.unlink()

    def clear_old_cache(self, days=7):
        """Clear cached results older than specified days."""
        import time
        temp_dir = Path(tempfile.gettempdir()) / "thz_deconvolution_cache"
        if temp_dir.exists():
            cutoff_time = time.time() - (days * 24 * 3600)
            for cache_file in temp_dir.glob("deconv_*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()

    def _resolve_times(self, times):
        if times is not None:
            return times
        if self._fitting_times is not None:
            return self._fitting_times
        raise ValueError(
            "`times` must be provided: no fitting-time axis is available (the PSF was "
            "loaded from psf_path rather than fitted from knife-edge measurements)."
        )

    def apply_deconvolution(self, scan, x_min, x_max, nx, y_min, y_max, ny, times=None, max_iter=500,
                             n_filters=25, start_freq=0.1, end_freq=10.0, win_width=0.5):
        times = self._resolve_times(times)
        scan_hash = hashlib.md5(scan.tobytes()).hexdigest()

        scan_params = {
            'scan_hash': scan_hash,
            'scan_shape': scan.shape,
            'x_min': x_min, 'x_max': x_max, 'nx': nx,
            'y_min': y_min, 'y_max': y_max, 'ny': ny,
            'max_iter': max_iter,
            'n_filters': n_filters, 'start_freq': start_freq, 'end_freq': end_freq, 'win_width': win_width,
        }
        config_hash = self._create_config_hash(scan_params)

        cached_result = self._load_cached_result(config_hash)
        if cached_result is not None:
            print(f"Loading cached deconvolution result (hash: {config_hash[:8]}...)")
            return cached_result

        filter_coeffs, center_freqs, _fs = _build_runtime_filter_bank(
            times, n_filters, start_freq, end_freq, win_width)

        dx = (x_max - x_min) / (nx - 1) if nx > 1 else (x_max - x_min)
        dy = (y_max - y_min) / (ny - 1) if ny > 1 else (y_max - y_min)

        wx_values = self.psf.wx_fit.evaluate(center_freqs)
        wy_values = self.psf.wy_fit.evaluate(center_freqs)
        w_min = min(np.min(wx_values), np.min(wy_values))
        w_max = max(np.max(wx_values), np.max(wy_values))

        img_nx, img_ny, _ = scan.shape
        max_allowed_x = (img_nx - 2) * dx / 2.0
        max_allowed_y = (img_ny - 2) * dy / 2.0

        deconvolved = np.zeros(scan.shape)
        flat_scan = scan.reshape(-1, scan.shape[-1])

        print("")
        print(f"Computing deconvolution (hash: {config_hash[:8]}...)")
        for nf in tqdm(range(n_filters)):
            center_freq = center_freqs[nf]
            wx, wy = wx_values[nf], wy_values[nf]
            x0 = self.psf.x0_spline.eval_const_extrap(center_freq)
            y0 = self.psf.y0_spline.eval_const_extrap(center_freq)

            range_max_x = range_max_min((wx + abs(x0)) * 3.0, 2.5)
            range_max_y = range_max_min((wy + abs(y0)) * 3.0, 2.5)
            range_max_x = (range_max_x // dx) * dx + dx
            range_max_y = (range_max_y // dy) * dy + dy
            range_max_x = min(range_max_x, max_allowed_x)
            range_max_y = min(range_max_y, max_allowed_y)

            nx_half = int(np.floor(range_max_x / dx))
            ny_half = int(np.floor(range_max_y / dy))
            x = np.arange(-nx_half, nx_half + 1) * dx
            y = np.arange(-ny_half, ny_half + 1) * dy

            gaussian_x = gaussian(x, x0, wx)
            gaussian_y = gaussian(y, y0, wy)
            _, _, psf_2d = create_psf_2d(gaussian_x, gaussian_y, x, y, dx, dy)

            traces_filtered = np.array([signal.convolve(trace, filter_coeffs[nf], mode='same')
                                         for trace in flat_scan]).reshape(scan.shape)

            image_filtered = np.sum(traces_filtered ** 2, axis=2) + 1.0  # Avoid division by zero

            n_iter = (max_iter if w_max == w_min
                      else int((wx - w_min) / (w_max - w_min) * (max_iter - 1) + 1))

            deconvolved_image = richardson_lucy_unclipped(image_filtered, psf_2d, num_iter=n_iter)
            deconvolved_image = np.maximum(deconvolved_image, 0.0)

            deconvolution_gains = np.sqrt(deconvolved_image / image_filtered)
            traces_filtered *= deconvolution_gains[:, :, None]
            deconvolved += traces_filtered

        self._save_cached_result(config_hash, deconvolved)
        return deconvolved

    def apply_single_pulse_deconvolution(self, pulse, times=None, max_iter=500,
                                          n_filters=25, start_freq=0.1, end_freq=10.0, win_width=0.5,
                                          dx=0.1, dy=0.1):
        """
        Apply deconvolution to a single pulse (no spatial grid, so `dx`/`dy`
        default to 0.1 mm — pass explicit values to match your scan's pitch).
        """
        times = self._resolve_times(times)
        filter_coeffs, center_freqs, _fs = _build_runtime_filter_bank(
            times, n_filters, start_freq, end_freq, win_width)

        wx_values = self.psf.wx_fit.evaluate(center_freqs)
        wy_values = self.psf.wy_fit.evaluate(center_freqs)
        w_min = min(np.min(wx_values), np.min(wy_values))
        w_max = max(np.max(wx_values), np.max(wy_values))

        deconvolved_pulse = np.zeros_like(pulse, dtype=np.float64)

        for nf in range(n_filters):
            pulse_filtered = signal.convolve(pulse, filter_coeffs[nf], mode='same')

            center_freq = center_freqs[nf]
            wx, wy = wx_values[nf], wy_values[nf]
            x0 = self.psf.x0_spline.eval_const_extrap(center_freq)
            y0 = self.psf.y0_spline.eval_const_extrap(center_freq)

            range_max_x = max((wx + abs(x0)) * 3.0, 2.5)
            range_max_y = max((wy + abs(y0)) * 3.0, 2.5)

            nx_half = int(np.floor(range_max_x / dx))
            ny_half = int(np.floor(range_max_y / dy))
            x = np.arange(-nx_half, nx_half + 1) * dx
            y = np.arange(-ny_half, ny_half + 1) * dy

            gaussian_x = gaussian(x, x0, wx)
            gaussian_y = gaussian(y, y0, wy)
            _, _, psf_2d = create_psf_2d(gaussian_x, gaussian_y, x, y, dx, dy)

            image_size = max(len(x), len(y))
            pulse_image = np.zeros((image_size, image_size))
            center = image_size // 2
            pulse_image[center, center] = np.sum(pulse_filtered ** 2) + 1.0

            n_iter = (max_iter if w_max == w_min
                      else int((wx - w_min) / (w_max - w_min) * (max_iter - 1) + 1))

            deconvolved_image = richardson_lucy_unclipped(pulse_image, psf_2d, num_iter=n_iter)
            deconvolved_image = np.maximum(deconvolved_image, 0.0)

            gain = np.sqrt(deconvolved_image[center, center] / pulse_image[center, center])
            deconvolved_pulse += pulse_filtered * gain

        return deconvolved_pulse
