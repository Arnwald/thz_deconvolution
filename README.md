# THz Deconvolution Library
[![PEP8](https://github.com/dotTHzTAG/pydotthz/actions/workflows/format.yml/badge.svg)](https://github.com/Arnwald/thz_deconvolution/actions/workflows/format.yml)
[![PyPI](https://img.shields.io/pypi/v/pydotthz?label=pypi%20package)](https://pypi.org/project/thz_deconvolution/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pydotthz)](https://pypi.org/project/thz_deconvolution/)

The THz Deconvolution Library is a Python package designed for advanced signal processing in THz time-domain spectroscopy (THz-TDS). This library provides tools to analyze and process THz signals, with a focus on beam profiling and deconvolution techniques to address frequency-dependent beam spreading effects.

Published in IEEE Transactions on Terahertz Science and Technology: [DOI: 10.1109/TTHZ.2024.3456789](https://doi.org/10.1109/TTHZ.2024.3456789)

```
A. Demion, L. L. Stöckli, N. Thomas and S. Zahno, "Frequency-Dependent Deconvolution for Enhanced THz-TDS Scans: Accounting for Beam Width Variations in Time Traces," in IEEE Transactions on Terahertz Science and Technology, vol. 15, no. 3, pp. 505-513, May 2025, doi: 10.1109/TTHZ.2025.3546756.
keywords: {Frequency measurement;Time-frequency analysis;Imaging;Fourier transforms;Finite impulse response filters;Deconvolution;Time-domain analysis;Terahertz radiation;Antenna measurements;Spatial resolution;Deconvolution;knife edge technique;Richardson–Lucy (RL);Terahertz time-domain spectroscopy (THz-TDS);Wiener},

```

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

To install the library:

```shell
pip install thz-deconvolution
```

## Usage

### Beam Width Fitting Example

To fit the beam widths from knife edge measurements, load in the measurements (.thz file) of the x and y axis:

```python
from thz_deconvolution.deconvolution import Deconvolution
from pathlib import Path

DeconvolutionFilter = Deconvolution(
    knife_edge_x_path=Path("psf_data/example_beam_width/measurement_x/1750085285.8557956_data.thz"),
    knife_edge_y_path=Path("psf_data/example_beam_width/measurement_y/1750163177.929295_data.thz")
)
```

or load a psf file:

```python
from thz_deconvolution.deconvolution import Deconvolution
from pathlib import Path

DeconvolutionFilter = Deconvolution(
    psf_path=Path("psf_data/example_beam_width/psf_file.thz")
)
```

Then deconvole the scan ($n_x \times n_y \times n_t$ data):

```python
    deconvolved_traces = DeconvolutionFilter.apply_deconvolution(scan, x_min, x_max, nx, y_min, y_max, ny, max_iter=100)
```

or deconvole a single waveform:

```python
    deconvolved_trace = DeconvolutionFilter.apply_deconvolution_single_trace(trace, max_iter=100)
```

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
- Implement tests

## License

This project is licensed under the MIT License. See the LICENSE file for details.
