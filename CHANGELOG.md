# THz Image Explorer changelog

All notable changes to the `thz-deconvolution` project will be documented in this file.

# 2.0.0 - Unreleased

**Breaking change**, matching the new PSF Tool in [thz-image-explorer](https://github.com/unibe-icelab/thz-image-explorer):

* Replaced the discrete per-filter PSF representation (`filters`/`filt_freqs`/`[x_0,w_x]`/`[y_0,w_y]` arrays) with a
  continuous model: a physical `a/f + b` fit plus a cubic-spline correction for beam widths (`wx`/`wy`), and a plain
  cubic spline for beam centers (`x0`/`y0`). `.npz` PSF files are now interchangeable with thz-image-explorer's
  "PSF Tool" in both directions — a PSF exported from the GUI tool can be loaded with `psf_path=`, and a PSF fitted
  here can be imported into the GUI tool.
* PSF fitting now processes the X and Y knife-edge measurement files independently (each split into left/right
  halves and averaged) instead of jointly, matching the GUI tool's pipeline. A single axis file is now also
  supported on its own (mirrored to both dimensions), previously both files were required.
* Old-format `.npz` PSF files (v1.x) can no longer be loaded — regenerate them from the knife-edge measurements.
  This matches thz-image-explorer itself, which dropped support for the old format too. Projects that need the old
  behavior can stay on `thz_deconvolution==1.1.1`.
* The deconvolution-time filter bank is now rebuilt from the scan's own time axis at `apply_deconvolution`/
  `apply_single_pulse_deconvolution` time (new `times=` parameter, falling back to the knife-edge measurement's time
  axis when the PSF was fitted here), decoupled from the filter bank used to fit the PSF. The PSF's 2D Gaussian grid
  now uses the scan's own pixel pitch (`dx`/`dy` derived from `x_min/x_max/nx` and `y_min/y_max/ny`) instead of the
  knife-edge measurement's point spacing.
* Default fitting-time frequency range changed from 0.25-4.0 THz to 0.15-5.0 THz to match the GUI tool's defaults.
* Dropped the `scan_type` ('reflectance'/'transmission') Gaussian flip — the GUI tool no longer distinguishes these.
* Removed the old discrete-filter functions (`create_filters`, `load_knife_edge_meas`, `fit_mean_beam`,
  `fit_beam_widths`, `create_psf_2d`, `richardson_lucy_freq`, `richardson_lucy_single_pulse`, `error_f`, `beam_w`)
  from `thz_deconvolution.utils`; equivalents now live in the new `psf`, `data_loader`, `filters`, and `fitting`
  modules. Generic utilities unrelated to the PSF model (FFT/windowing helpers, `richardson_lucy(_unclipped)`,
  `wiener_freq`/`wiener_worker`) are unchanged.
* The `Deconvolution` constructor and `apply_deconvolution`/`apply_single_pulse_deconvolution` keep the same
  call shape as 1.x (same positional/keyword arguments), aside from the additions above.

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