"""Continuous PSF model: hybrid physical+spline fit for beam widths, plain
cubic splines for beam centers.

This mirrors thz-image-explorer's `src/filters/psf.rs` (the struct actually
used at deconvolution time and round-tripped through `.npz`) and
`src/psf_tool/curve_fitting.rs` (how it's fitted from per-filter beam-width
data). PSF files produced here are interchangeable with the "PSF Tool" in the
thz-image-explorer GUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate

__all__ = ["gaussian", "create_psf_2d", "CubicSpline", "HybridFit", "PSF"]


def gaussian(x, x0, w):
    x = np.asarray(x, dtype=np.float64)
    return math.sqrt(2.0 / math.pi) * np.exp(-2.0 * (x - x0) ** 2 / (w ** 2)) / w


def create_psf_2d(psf_x, psf_y, x, y, dx, dy):
    """Build a 2D PSF grid at spacing (dx, dy) by interpolating 1D beam
    profiles, zero-padded to twice their extent for a clean interpolation.

    dx/dy are the *output* grid step (e.g. the scanned image's pixel pitch),
    independent of the input arrays' own sampling.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    psf_x = np.array(psf_x, dtype=np.float64)
    psf_y = np.array(psf_y, dtype=np.float64)
    psf_x = psf_x / np.max(psf_x)
    psf_y = psf_y / np.max(psf_y)

    x_max = math.floor(np.max(x))
    y_max = math.floor(np.max(y))

    factor = 2.0
    new_x_max = factor * x_max
    new_y_max = factor * y_max

    x_step = x[-1] - x[-2]
    y_step = y[-1] - y[-2]
    n_new_steps_x = int(np.ceil((new_x_max - x[-1]) / x_step))
    n_new_steps_y = int(np.ceil((new_y_max - y[-1]) / y_step))

    for _ in range(n_new_steps_x):
        x = np.append(x, x[-1] + x_step)
        x = np.append(x[0] - x_step, x)
        psf_x = np.append(psf_x, 0.0)
        psf_x = np.append(0.0, psf_x)
    for _ in range(n_new_steps_y):
        y = np.append(y, y[-1] + y_step)
        y = np.append(y[0] - y_step, y)
        psf_y = np.append(psf_y, 0.0)
        psf_y = np.append(0.0, psf_y)

    xx = np.arange(-x_max, x_max + dx, dx)
    yy = np.arange(-y_max, y_max + dy, dy)
    X, Y = np.meshgrid(xx, yy)

    psfx = interpolate.interp1d(x, psf_x, kind='slinear', fill_value='extrapolate')
    psfy = interpolate.interp1d(y, psf_y, kind='slinear', fill_value='extrapolate')

    psf_2d = psfx(X) * psfy(Y)
    return X, Y, psf_2d


def _solve_tridiagonal(a, b, c, d):
    n = len(b)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


@dataclass
class CubicSpline:
    """Natural cubic spline: x/y are the knots, coeffs[i] = [a,b,c,d] for
    segment i, S_i(dx) = a + b*dx + c*dx^2 + d*dx^3 with dx = t - x[i]."""
    x: np.ndarray
    y: np.ndarray
    coeffs: np.ndarray  # shape (n-1, 4)

    @classmethod
    def fit(cls, x, y) -> "CubicSpline":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for a spline")

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        n = len(x_sorted)

        h = np.diff(x_sorted)
        if np.any(h <= 0.0):
            raise ValueError("x values must be strictly increasing")

        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        d = np.zeros(n)
        b[0] = 1.0
        b[n - 1] = 1.0
        for i in range(1, n - 1):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 3.0 * ((y_sorted[i + 1] - y_sorted[i]) / h[i]
                          - (y_sorted[i] - y_sorted[i - 1]) / h[i - 1])

        m = _solve_tridiagonal(a, b, c, d)

        coeffs = np.zeros((n - 1, 4))
        for i in range(n - 1):
            dx = h[i]
            dy = y_sorted[i + 1] - y_sorted[i]
            coeffs[i, 0] = y_sorted[i]
            coeffs[i, 1] = dy / dx - dx * (2.0 * m[i] + m[i + 1]) / 3.0
            coeffs[i, 2] = m[i]
            coeffs[i, 3] = (m[i + 1] - m[i]) / (3.0 * dx)

        return cls(x=x_sorted, y=y_sorted, coeffs=coeffs)

    def _eval_interior(self, xv: float) -> float:
        """Evaluate the polynomial, assuming x[0] <= xv <= x[-1]."""
        n = len(self.x)
        left, right = 0, n - 1
        while right - left > 1:
            mid = (left + right) // 2
            if self.x[mid] > xv:
                right = mid
            else:
                left = mid
        dx = xv - self.x[left]
        c = self.coeffs[left]
        return c[0] + c[1] * dx + c[2] * dx * dx + c[3] * dx * dx * dx

    def eval_const_extrap(self, xv: float) -> float:
        """Evaluate with constant extrapolation (holds boundary values) —
        used for beam-center (x0/y0) splines."""
        if xv < self.x[0]:
            return float(self.y[0])
        if xv > self.x[-1]:
            return float(self.y[-1])
        return self._eval_interior(xv)

    def eval_array_const_extrap(self, xs) -> np.ndarray:
        return np.array([self.eval_const_extrap(v) for v in np.asarray(xs, dtype=np.float64)])


@dataclass
class HybridFit:
    """Physical base model a/f + b, plus a cubic-spline correction fitted to
    the residuals — used for beam widths (wx/wy)."""
    a: float
    b: float
    correction: CubicSpline

    @classmethod
    def fit(cls, frequencies, values) -> "HybridFit":
        frequencies = np.asarray(frequencies, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if len(frequencies) != len(values):
            raise ValueError("frequencies and values must have the same length")
        if len(frequencies) < 3:
            raise ValueError("Need at least 3 points for a hybrid fit")

        inv_f = 1.0 / frequencies
        sum_1_f = np.sum(inv_f)
        sum_1_f2 = np.sum(inv_f * inv_f)
        sum_w = np.sum(values)
        sum_w_f = np.sum(values * inv_f)
        n = float(len(frequencies))

        det = sum_1_f2 * n - sum_1_f * sum_1_f
        if abs(det) < 1e-10:
            raise ValueError("Singular matrix in base fit")

        a = (sum_w_f * n - sum_w * sum_1_f) / det
        b = (sum_1_f2 * sum_w - sum_1_f * sum_w_f) / det

        residuals = values - (a / frequencies + b)
        correction = CubicSpline.fit(frequencies, residuals)

        return cls(a=a, b=b, correction=correction)

    def _eval_correction(self, f: float) -> float:
        corr = self.correction
        n = len(corr.x)
        f_min, f_max = corr.x[0], corr.x[-1]

        if f_min <= f <= f_max:
            return corr._eval_interior(f)

        if f < f_min:
            dx = f - f_min
            y0, slope = corr.coeffs[0][0], corr.coeffs[0][1]
            max_slope = self.a / (f * f)
            safe_slope = min(slope, max_slope)
            return y0 + safe_slope * dx
        else:
            i = n - 2
            dx_end = corr.x[-1] - corr.x[i]
            c = corr.coeffs[i]
            y_end = c[0] + c[1] * dx_end + c[2] * dx_end ** 2 + c[3] * dx_end ** 3
            slope_end = c[1] + 2.0 * c[2] * dx_end + 3.0 * c[3] * dx_end ** 2
            max_slope = self.a / (f * f)
            safe_slope = min(slope_end, max_slope)
            dx = f - corr.x[-1]
            return y_end + safe_slope * dx

    def eval_single(self, f: float) -> float:
        base = self.a / f + self.b
        return max(base + self._eval_correction(f), 1e-6)

    def evaluate(self, freqs) -> np.ndarray:
        return np.array([self.eval_single(f) for f in np.asarray(freqs, dtype=np.float64)])


_SPLINE_SUFFIXES = ("knots_thz", "values_mm", "coeff_a", "coeff_b", "coeff_c", "coeff_d")


def _spline_to_npz(prefix: str, spline: CubicSpline) -> dict:
    return {
        f"{prefix}_knots_thz": spline.x.astype(np.float64),
        f"{prefix}_values_mm": spline.y.astype(np.float64),
        f"{prefix}_coeff_a": spline.coeffs[:, 0].astype(np.float64),
        f"{prefix}_coeff_b": spline.coeffs[:, 1].astype(np.float64),
        f"{prefix}_coeff_c": spline.coeffs[:, 2].astype(np.float64),
        f"{prefix}_coeff_d": spline.coeffs[:, 3].astype(np.float64),
    }


def _spline_from_npz(data, prefix: str) -> CubicSpline:
    return CubicSpline(
        x=np.asarray(data[f"{prefix}_knots_thz"], dtype=np.float64).reshape(-1),
        y=np.asarray(data[f"{prefix}_values_mm"], dtype=np.float64).reshape(-1),
        coeffs=np.stack(
            [np.asarray(data[f"{prefix}_coeff_{c}"], dtype=np.float64).reshape(-1)
             for c in "abcd"],
            axis=1,
        ),
    )


@dataclass
class PSF:
    """The full PSF: hybrid fits for beam width, cubic splines for beam
    center, both as continuous functions of frequency. Matches the `.npz`
    schema written/read by thz-image-explorer's `src/io.rs::load_psf` and
    `src/psf_tool/export.rs::export_to_npz`."""
    wx_fit: HybridFit
    wy_fit: HybridFit
    x0_spline: CubicSpline
    y0_spline: CubicSpline

    @classmethod
    def fit_from_data(cls, frequencies, wx, wy, x0, y0) -> "PSF":
        return cls(
            wx_fit=HybridFit.fit(frequencies, wx),
            wy_fit=HybridFit.fit(frequencies, wy),
            x0_spline=CubicSpline.fit(frequencies, x0),
            y0_spline=CubicSpline.fit(frequencies, y0),
        )

    def save_npz(self, path) -> None:
        data = {
            "wx_base_a": np.array([self.wx_fit.a], dtype=np.float64),
            "wx_base_b": np.array([self.wx_fit.b], dtype=np.float64),
            "wy_base_a": np.array([self.wy_fit.a], dtype=np.float64),
            "wy_base_b": np.array([self.wy_fit.b], dtype=np.float64),
        }
        data.update(_spline_to_npz("wx_corr", self.wx_fit.correction))
        data.update(_spline_to_npz("wy_corr", self.wy_fit.correction))
        data.update(_spline_to_npz("x0", self.x0_spline))
        data.update(_spline_to_npz("y0", self.y0_spline))
        np.savez(Path(path), **data)

    @classmethod
    def load_npz(cls, path) -> "PSF":
        with np.load(Path(path)) as data:
            wx_a = float(np.asarray(data["wx_base_a"]).reshape(-1)[0])
            wx_b = float(np.asarray(data["wx_base_b"]).reshape(-1)[0])
            wy_a = float(np.asarray(data["wy_base_a"]).reshape(-1)[0])
            wy_b = float(np.asarray(data["wy_base_b"]).reshape(-1)[0])
            wx_fit = HybridFit(a=wx_a, b=wx_b, correction=_spline_from_npz(data, "wx_corr"))
            wy_fit = HybridFit(a=wy_a, b=wy_b, correction=_spline_from_npz(data, "wy_corr"))
            x0_spline = _spline_from_npz(data, "x0")
            y0_spline = _spline_from_npz(data, "y0")
        return cls(wx_fit=wx_fit, wy_fit=wy_fit, x0_spline=x0_spline, y0_spline=y0_spline)
