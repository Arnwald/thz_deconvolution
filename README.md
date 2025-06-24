# THz Deconvolution Library

The THz Deconvolution Library is a Python package designed for advanced signal processing in THz time-domain spectroscopy (THz-TDS). This library provides tools to analyze and process THz signals, with a focus on beam profiling and deconvolution techniques to address frequency-dependent beam spreading effects.

## Overview

THz time-domain spectroscopy is a powerful tool for studying materials and systems in the terahertz frequency range. However, the analysis of THz signals is often complicated by frequency-dependent beam spreading and other distortions. This library aims to simplify and enhance the processing of THz signals by providing robust algorithms and utilities for beam profiling, deconvolution, and signal restoration.

## Key Features

### 1. Beam Width Fitting

- **Knife Edge Measurements**: Fit beam widths using knife edge measurement data to accurately characterize the THz beam.
- **Frequency-Dependent Profiling**: Profile the beam width per frequency to account for low-frequency spreading effects, ensuring precise analysis across the spectrum.
- **Gaussian Beam Fitting**: Fit Gaussian profiles to beam data for accurate modeling and parameter extraction.

### 2. Deconvolution Algorithms

- **Classical Richardson-Lucy Deconvolution**:
  - Supports clipped and unclipped variants
- **Frequency-dependent Richardson-Lucy Deconvolution**:
  - Accounts for frequency-dependent distortions in the beam profile.
  - The time traces of the scans are modified to account for frequency-dependent distortions, thus preserving the depth information.
- **Frequency-dependent Wiener Deconvolution**:
  - Noise-robust signal restoration using Wiener filtering.
  - The time traces of the scans are modified to account for frequency-dependent distortions, thus preserving the depth information.

### 3. Signal Processing Utilities

- **Windowing and Zero Padding**: Tools for preparing signals for Fourier analysis, including Blackman and Kaiser windows.
- **Bandpass Filtering**: Apply Kaiser window-based bandpass filters to isolate specific frequency ranges.
- **FFT Utilities**: Efficient computation of FFTs and related operations, including zero-padded FFTs for enhanced resolution.
- **Custom Filters**: Create and apply custom filters tailored to specific signal processing needs.

### 4. Data Handling

- **Knife Edge Data Loading**: Load and preprocess knife edge measurement data for beam profiling.
- **Signal Windowing**: Extract and process specific signal regions using custom window functions.

## Installation

To install the library and its dependencies, clone the repository and use the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually using `pip`:

```bash
pip install numpy matplotlib scikit-image tqdm multiprocess
pip install git+https://github.com/dotTHzTAG/pydotthz.git@main#egg=pydotthz
```

## Usage

### Beam Width Fitting Example

```python
from thz_deconvolution import load_knife_edge_meas, fit_mean_beam, fit_beam_widths, create_psf_2d, create_filters

# Frequency and filter parameters
low_cut = 0.1
high_cut = 10.0
start_freq = 0.25
end_freq = 4.0
win_width = 0.5
n_filters = 20
w_max = 30

# Load knife edge measurement data
x_axis, y_axis, psf_t_x, psf_t_y, times_psf  = load_knife_edge_meas("path_to_measurement_file_x", "path_to_measurement_file_y")

# Centering the data
x_axis -= np.mean(x_axis)
y_axis -= np.mean(y_axis)

print("Fitting the mean PSF")
x0, y0, popt_x, popt_y = fit_mean_beam(
    x_axis, y_axis, psf_t_x, psf_t_y)

# Create the PSF
x_start = np.abs(x_axis[0])
y_start = np.abs(y_axis[0])
dx = np.abs(x_axis[1] - x_axis[0])
dy = np.abs(y_axis[1] - y_axis[0])
xx = np.arange(-x_start, x_start + dx, dx)
yy = np.arange(-y_start, y_start + dy, dy)

gauss_x = gaussian(xx, 0.0, popt_x[1])
gauss_y = gaussian(yy, 0.0, popt_y[1])
gauss_x = gauss_x / np.max(gauss_x)
gauss_y = gauss_y / np.max(gauss_y)

_, _, psf_2d = create_psf_2d(gauss_x, gauss_y, xx, yy)

print("Creating the filters for the PSF")
filters, filt_freqs = create_filters(
    n_filters, times_psf, win_width, low_cut, high_cut, start_freq, end_freq)

print("Fitting the PSF beam widths by frequency")
_, _, popt_xs, popt_ys, _, _ = fit_beam_widths(
    x0, y0, x_psf, y_psf, np_psf_t_x, np_psf_t_y, filters, filt_freqs, w_max)
```

`popt_xs` and `popt_ys` contain all the parameters necessary to fit a gaussian beam at each frequency.

### Deconvolution Example

```python
from thz_deconvolution import richardson_lucy_freq

# Initialize the maximum number of iterations for Richardson-Lucy deconvolution
max_iter = 500

meas_type == 'reflectance' # or meas_type == 'transmission', to mirror the PSF
# Perform Richardson-Lucy deconvolution in the frequency domain
deconvolved_traces = richardson_lucy_freq(scan, xx, yy, popt_xs, popt_ys, filters, filt_freqs, max_iter, scan_type=meas_type)
```

`deconvolved_traces` and `scan` have the same shape and dimensions. `scan` is the original data, containing time traces for each (x,y) points in the scan.

## Dependencies

This library relies on the following Python packages:

- `numpy`: For numerical computations.
- `matplotlib`: For data visualization.
- `scikit-image`: For image and signal processing.
- `tqdm`: For progress bars in iterative algorithms.
- `multiprocess`: For parallel processing of computationally intensive tasks.
- `pydotthz`: A custom library for THz signal processing (installed via GitHub).

Ensure all dependencies are installed by running the provided `requirements.txt`.

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to:

1. Open an issue to discuss your ideas.
2. Submit a pull request with your changes.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## Acknowledgments

This library was developed to support research in THz time-domain spectroscopy. Special thanks to the contributors and the open-source community for their invaluable tools and resources. This work is part of the MARVIS-Subice research program: [https://subice.unibe.ch](https://subice.unibe.ch).

## References

1. [Richardson-Lucy Deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)
2. [Wiener Filtering](https://en.wikipedia.org/wiki/Wiener_filter)
3. [THz Time-Domain Spectroscopy](https://en.wikipedia.org/wiki/Terahertz_time-domain_spectroscopy)

## Future Work

- Add support for additional deconvolution algorithms.
- Expand the library to include more advanced beam profiling techniques.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
