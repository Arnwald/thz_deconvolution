# THz Image Explorer changelog

All notable changes to the `thz-deconvolution` project will be documented in this file.

# Unreleased 1.1.X - X.X.2025

* Documented the Git LFS requirement for fetching example data (`psf_data/`, `sample_data/`) in the README
* Fixed README usage examples referencing the non-existent `apply_deconvolution_single_trace` method (now `apply_single_pulse_deconvolution`) and a wrong `.thz` extension for the PSF file example (now `.npz`)
* Fixed a stale placeholder DOI in the README's publication badge
* Added missing `scipy` entry to the README's Dependencies list
* Added titles, axis labels, and a legend to the figures in `examples/simple.py`

# 1.1.0 - 17.10.2025
* Added example scripts for beam width fitting and deconvolution in the README.md
* Added Deconvolution class that tracks hashed filter settings/deconvolved traces
* Added example data with git LFS
* Added citation file