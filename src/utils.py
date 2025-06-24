from scipy.signal.windows import tukey
from scipy.signal import convolve
from scipy import interpolate
from skimage import restoration
from scipy.fft import rfft, irfft, rfftfreq
from tqdm import tqdm
from scipy.signal import firwin, kaiser_atten, kaiser_beta, freqz
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import signal
from pydotthz import DotthzFile
import scipy.special
import multiprocess as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import math

# Create a 2D PSF from 2x1D PSFs
def create_psf_2d(psf_x, psf_y, x, y, plot=False):
    """
    Create a 2D Point Spread Function (PSF) from 1D PSFs along x and y axes.

    Parameters
    ----------
    psf_x : array_like
        1D PSF along the x-axis.
    psf_y : array_like
        1D PSF along the y-axis.
    x : array_like
        x-axis positions.
    y : array_like
        y-axis positions.
    plot : bool, optional
        If True, plots the 2D PSF. Default is False.

    Returns
    -------
    X : ndarray
        Meshgrid of x-axis positions.
    Y : ndarray
        Meshgrid of y-axis positions.
    psf_2d : ndarray
        2D PSF created from the input 1D PSFs.
    """
    x_max = math.floor(np.max(x))
    y_max = math.floor(np.max(y))
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Creating the PSF grid
    X, Y = np.meshgrid(x, y)

    factor = 2.0
    # Padding the PSF with zeros for interpolation
    new_x_max = factor * x_max
    new_y_max = factor * y_max

    # Determining the number of new steps
    x_step = x[-1] - x[-2]
    N_new_steps_x = int((new_x_max - x[-1]) / x_step)

    y_step = y[-1] - y[-2]
    N_new_steps_y = int((new_y_max - y[-1]) / y_step)

    # Adding zeros to the PSF
    for _ in range(N_new_steps_x):
        x = np.append(x, x[-1] + x_step)
        x = np.append(x[0] - x_step, x)
        psf_x = np.append(psf_x, 0.0)
        psf_x = np.append(0.0, psf_x)
    for _ in range(N_new_steps_y):
        y = np.append(y, y[-1] + y_step)
        y = np.append(y[0] - y_step, y)
        psf_y = np.append(psf_y, 0.0)
        psf_y = np.append(0.0, psf_y)

    xx = np.arange(-x_max, x_max + dx, dx)
    yy = np.arange(-y_max, y_max + dy, dy)
    # Creating the PSF grid
    X, Y = np.meshgrid(xx, yy)

    # Interpolating the PSF
    psfx = interpolate.interp1d(x, psf_x, kind='slinear')
    psfy = interpolate.interp1d(y, psf_y, kind='slinear')

    psf_2d = psfx(X) * psfy(Y)

    if plot:
        plt.figure()
        plot = plt.pcolormesh(X, Y, psf_2d, cmap='jet')
        plt.xlabel("Position x [mm]")
        plt.ylabel("Position y [mm]")
        plt.title("PSF")
        plt.gca().set_aspect('equal')
        plt.show()

    return X, Y, psf_2d


# Richardson-Lucy deconvolution
# The Richardson-Lucy algorithm is an iterative algorithm that can be used to deconvolve an image blurred by a PSF.
# An integration exists in the skimage library, but we will implement it ourselves to better understand the algorithm.
# To use skimage's implementation, use the following code:
# from skimage import restoration
# deconvolved = restoration.richardson_lucy(image, PSF, iterations=100)
def richardson_lucy(d, psf, num_iter):
    """
    Perform Richardson-Lucy deconvolution on input data.

    Parameters
    ----------
    d : ndarray
        Input data to be deconvolved.
    psf : ndarray
        Point Spread Function (PSF) used for deconvolution.
    num_iter : int
        Number of iterations for the deconvolution process.

    Returns
    -------
    ndarray
        Deconvolved data.
    """
    # Padding the data to avoid edge effects
    pad = int(psf.shape[0] / 8)
    d = np.pad(d, pad, 'minimum')
    psf = psf / np.sum(psf)
    psf_T = np.flip(psf) # Flipped PSF
    u = d.copy() # Initial guess
    eps = 1e-12 # Regularization parameter to avoid division by zero
    for _ in range(num_iter):
        u = np.multiply(u, convolve(d / (convolve(u, psf, mode='same') + eps), psf_T, mode='same'))
    # Clipping the values
    u = np.clip(u, 0, 1)
    return u[pad:-pad, pad:-pad]


# Richardson-Lucy deconvolution
# The Richardson-Lucy algorithm is an iterative algorithm that can be used to deconvolve an image blurred by a PSF.
# An integration exists in the skimage library, but we will implement it ourselves to better understand the algorithm.
# To use skimage's implementation, use the following code:
# from skimage import restoration
# deconvolved = restoration.richardson_lucy(image, PSF, num_iter=100)
def richardson_lucy_unclipped(d, psf, num_iter):
    # Padding the data to avoid edge effects
    pad = int(psf.shape[0] / 2)
    d = np.pad(d, pad, 'reflect')
    psf = psf / np.sum(psf)
    psf_T = np.flip(psf) # Flipped PSF
    u = d.copy() # Initial guess
    eps = 1e-12 # Regularization parameter to avoid division by zero
    for _ in range(num_iter):
        u = np.multiply(u, convolve(d / (convolve(u, psf, mode='same') + eps), psf_T, mode='same'))
    return u[pad:-pad, pad:-pad]


# Blackman window
def blackman_func(n, M):
    """
    Compute the Blackman window function.

    Parameters
    ----------
    n : array_like
        Input array.
    M : float
        Window length.

    Returns
    -------
    array_like
        Blackman window values.
    """
    return 0.42 - 0.5 * np.cos(2 * np.pi * n / M) + 0.08 * np.cos(4 * np.pi * n / M)

# Reproduction of the toptica window function
def toptica_window(t, start=1, end=7):
    """
    Apply a Toptica window function to the input time array.

    Parameters
    ----------
    t : array_like
        Time array.
    start : float, optional
        Start time for the window. Default is 1.
    end : float, optional
        End time for the window. Default is 7.

    Returns
    -------
    ndarray
        Windowed time array.
    """
    window = np.ones(t.shape)
    a = t[t <= (t[0] + start)]
    b = t[t >= (t[-1] - end)]
    a = blackman_func(a - a[0], 2 * (a[-1] - a[0]))
    b = blackman_func(b + b[-1] - b[0] - b[0], 2 * (b[-1] - b[0]))
    window[t <= (t[0] + start)] = a
    window[t >= (t[-1] - end)] = b
    return window

# Zero-padding function to extend the time array
def zero_padding(time, pulse, df_padded=0.01):
    """
    Apply zero-padding to a signal to achieve a desired frequency resolution.

    Parameters
    ----------
    time : array_like
        Time array of the signal.
    pulse : array_like
        Signal data.
    df_padded : float, optional
        Desired frequency resolution. Default is 0.01.

    Returns
    -------
    extended_time : ndarray
        Extended time array after zero-padding.
    padded_pulse : ndarray
        Zero-padded signal.
    """
    # Calculate the total time span of the original data
    T = time[-1] - time[0]

    # Determine the required number of points to achieve the desired frequency resolution
    N_padded = int(np.ceil(T / df_padded))

    # Find the length of the original signal
    N_original = len(pulse)

    # Calculate the original time step (assuming uniform sampling in the time array)
    dt = time[1] - time[0]

    # If padding is needed, apply zero-padding and extend the time array
    if N_padded > N_original:
        # Pad the pulse array with zeros to match the required length
        padded_pulse = np.pad(pulse, (0, N_padded - N_original), mode='constant')

        # Create an extended time array with the same timestep (dt)
        extended_time = np.arange(time[0], time[0] + N_padded * dt, dt)
    else:
        # If no padding is needed, return the original arrays
        padded_pulse = pulse
        extended_time = time

    return extended_time, padded_pulse


def get_fft(t, p, df=0.01, window_start=1, window_end=7, return_td=False):
    """
    Compute the FFT of a signal with optional windowing and zero-padding.

    Parameters
    ----------
    t : array_like
        Time array of the signal.
    p : array_like
        Signal data.
    df : float, optional
        Desired frequency resolution. Default is 0.01.
    window_start : float, optional
        Start time for the window. Default is 1.
    window_end : float, optional
        End time for the window. Default is 7.
    return_td : bool, optional
        If True, returns time-domain data along with FFT. Default is False.

    Returns
    -------
    f : ndarray
        Frequency array.
    a : ndarray
        Amplitude spectrum.
    arg : ndarray
        Phase spectrum.
    """
    t = np.array(t)
    p = np.array(p) * toptica_window(t, window_start, window_end)
    t, p = zero_padding(t, p, df_padded=df)

    sample_rate = 1 / (t[1] - t[0]) * 1e12
    n = len(p)
    fft = rfft(p)
    a = np.abs(fft)
    angle = np.angle(fft)
    arg = np.unwrap(angle)
    f = rfftfreq(n, 1 / sample_rate) / 1e12
    if return_td:
        return t,p,f, a, np.abs(arg)
    else:
        return f, a, np.abs(arg)


def get_fft_c(t, p, df=0.01, window_start=1, window_end=7):
    t = np.array(t)
    p = np.array(p) * toptica_window(t, window_start, window_end)
    t, p = zero_padding(t, p, df_padded=df)

    fft_p = rfft(p)
    return fft_p

# Extracting a subtring from a text between two strings
def extract_substring(text, start_str, end_str):
    try:
        # Find the start and end indices of the substring
        start_idx = text.index(start_str) + len(start_str)
        if end_str == "":
            end_idx = len(text)
        else:
            end_idx = text.index(end_str, start_idx)
        # Extract the substring
        substring = text[start_idx:end_idx]

        return substring
    except ValueError as e:
        print(e)
        # Return None if the start_str or end_str are not found
        return None

# Error function
def error_f(x, x0, w):
    """
    Compute the error function for fitting purposes.

    Parameters
    ----------
    x : array_like
        Input data points.
    x0 : float
        Center of the error function.
    w : float
        Width parameter of the error function.

    Returns
    -------
    ndarray
        Computed error function values.
    """
    return (1 + scipy.special.erf(math.sqrt(2) * (x - np.array(x0)) / w)) / 2

# Gaussian function
def gaussian(x, x0, w):
    return math.sqrt(2 / math.pi) * np.exp(-2 * (x - np.array(x0)) ** 2 / (w ** 2)) / w

# Expected beam width function
def beam_w(freq, a):
    """
    Compute the beam width as a function of frequency.

    Parameters
    ----------
    freq : array_like
        Frequencies at which to compute the beam width.
    a : float
        Scaling factor for the beam width.

    Returns
    -------
    ndarray
        Beam width values corresponding to the input frequencies.
    """
    return a / np.array(freq)

# Kaiser windowed FIR filter
def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    atten = kaiser_atten(ntaps, width/(0.5*fs))
    beta = kaiser_beta(atten)
    if lowcut <= 0.0:
        cutoffs = highcut
        pass_zero = 'lowpass'
    elif highcut >= 0.5*fs:
        cutoffs = lowcut
        pass_zero = 'highpass'
    else:
        cutoffs = [lowcut, highcut]
        pass_zero = 'bandpass'
    taps = firwin(ntaps, cutoffs, fs=fs, pass_zero=pass_zero,
                  window=('kaiser', beta), scale=False)
    return taps

# Zero padding a signal left or right
def zero_pad(y, N_pad, lr='right'):
    """
    Zero padding.

    Adds zeros to the signal to make its total length equal to N_pad.

    Parameters
    ----------
    y : array_like
        Input signal to be padded.
    N_pad : int
        Desired total length of the padded signal.
    lr : {'right', 'left'}, optional
        Specifies whether to pad on the right or left. Default is 'right'.

    Returns
    -------
    array_like
        Zero-padded signal.
    """
    N = len(y)
    if N_pad > N:
        if lr == 'right':
            y = np.append(y, np.zeros(N_pad - N))
        elif lr == 'left':
            y = np.append(np.zeros(N_pad - N), y)
        else:
            raise ValueError("lr must be either 'right' or 'left'")
    return y

# Windowed signal
def get_windowed_signal(y, ratio=0.5, lr='right', window='tukey', alpha=0.02):
    """
    Apply a window to the signal with optional zero-padding.

    Parameters
    ----------
    y : array_like
        Input signal.
    ratio : float, optional
        Ratio of the window length to the signal length. Default is 0.5.
    lr : {'right', 'left'}, optional
        Specifies whether to pad on the right or left. Default is 'right'.
    window : {'tukey', 'boxcar'}, optional
        Type of window to apply. Default is 'tukey'.
    alpha : float, optional
        Shape parameter for the Tukey window. Default is 0.02.

    Returns
    -------
    tuple
        A tuple containing the windowed signal and the applied window.
    """
    N_win = int(len(y) * ratio)
    if window == 'tukey':
        w = tukey(N_win, alpha)
    elif window == 'boxcar':
        w = signal.filters.boxcar(N_win)
    else:
        raise ValueError("Window must be 'kaiser'")
    w = zero_pad(w, len(y), lr=lr)
    return y * w, w

# This function loads knife edge measurements from thz files.
def load_knife_edge_meas(x_path, y_path):
    """
    Load knife-edge measurement data from files.

    Parameters
    ----------
    x_path : str
        Path to the file containing x-axis measurements.
    y_path : str
        Path to the file containing y-axis measurements.

    Returns
    -------
    tuple
        A tuple containing:
        - pos_x : ndarray
            Positions along the x-axis.
        - pos_y : ndarray
            Positions along the y-axis.
        - np_psf_t_x_0 : ndarray
            PSF traces along the x-axis.
        - np_psf_t_y_0 : ndarray
            PSF traces along the y-axis.
        - times : ndarray
            Time array corresponding to the measurements.
    """

    psf_t_x = []
    psf_t_y = []
    pos_x = []
    pos_y = []
    xx = []
    yy = []
    len_traces = None
    times = None
    nx = 0
    ny = 0

    if not os.path.exists(x_path):
        print("Knife edge measurements in x not found. Exiting.")
        exit(1)

    if not os.path.exists(y_path):
        print("Knife edge measurements in y not found. Exiting.")
        exit(1)

    print("Loading x raw measurements...")
    with DotthzFile(x_path, "r") as file:
        # read the first group/measurement
        keys = list(file.measurements.keys())

        for key in keys:
            x = extract_substring(key, "=", "")
            datasets = file.measurements.get(key).datasets
            key = list(datasets.keys())[0]

            pulse_trace = np.array(datasets.get(key))[:, 1]

            pos_x.append(float(x))
            if times is None:
                times = np.array(datasets.get(key))[:, 0]
            if len_traces is None:
                len_traces = len(np.array(datasets.get(key))[:, 1])

            psf_t_x.append(pulse_trace)

    print("Loading y raw measurements...")
    with DotthzFile(y_path, "r") as file:
        # read the first group/measurement
        keys = list(file.measurements.keys())

        for key in keys:
            y = extract_substring(key, "=", "")
            datasets = file.measurements.get(key).datasets
            key = list(datasets.keys())[0]

            pulse_trace = np.array(datasets.get(key))[:, 1]

            pos_y.append(float(y))
            if times is None:
                times = np.array(datasets.get(key))[:, 0]
            if len_traces is None:
                len_traces = len(np.array(datasets.get(key))[:, 1])

            psf_t_y.append(pulse_trace)

    np_psf_t_x_0 = np.array(psf_t_x)
    np_psf_t_y_0 = np.array(psf_t_y)

    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)

    # Sorting by x positions
    inds = pos_x.argsort()
    pos_x = pos_x[inds]
    np_psf_t_x_0 = np_psf_t_x_0[inds]

    # Sorting by y positions
    inds = pos_y.argsort()
    pos_y = pos_y[inds]
    np_psf_t_y_0 = np_psf_t_y_0[inds]

    return pos_x, pos_y, np_psf_t_x_0, np_psf_t_y_0, times

# Get the center of the PSF
def fit_mean_beam(x_axis_psf, y_axis_psf, np_psf_t_x, np_psf_t_y, nrange=None, plot=False):
    """
    Fit the mean beam profile to the PSF data.

    Parameters
    ----------
    x_axis_psf : array_like
        x-axis positions of the PSF.
    y_axis_psf : array_like
        y-axis positions of the PSF.
    np_psf_t_x : ndarray
        PSF traces along the x-axis.
    np_psf_t_y : ndarray
        PSF traces along the y-axis.
    nrange : tuple of int, optional
        Range of indices to crop the PSF for fitting. Default is None.
    plot : bool, optional
        If True, plots the fitting results. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - x0 : float
            Center position of the PSF along the x-axis.
        - y0 : float
            Center position of the PSF along the y-axis.
        - popt_x : ndarray
            Optimal parameters for the x-axis fit.
        - popt_y : ndarray
            Optimal parameters for the y-axis fit.
    """

    print("Extracting the center of the PSF...")

    if nrange is not None:
        n_min = nrange[0]
        n_max = nrange[1]

        # Cropping the PSF to improve the fit
        # as the right part of the measurement is noisy
        # and presents deviations from the error function
        x_axis_psf_2 = x_axis_psf[n_min:n_max]
        y_axis_psf_2 = y_axis_psf[n_min:n_max]
        np_psf_t_x_2 = np_psf_t_x[n_min:n_max, :]
        np_psf_t_y_2 = np_psf_t_y[n_min:n_max, :]
    else:
        x_axis_psf_2 = x_axis_psf
        y_axis_psf_2 = y_axis_psf
        np_psf_t_x_2 = np_psf_t_x
        np_psf_t_y_2 = np_psf_t_y

    intensity_x = np.sum(np_psf_t_x[:, :] ** 2, axis=1)
    intensity_x = intensity_x - np.min(intensity_x)
    intensity_x = intensity_x / np.max(intensity_x)
    intensity_y = np.sum(np_psf_t_y[:, :] ** 2, axis=1)
    intensity_y = intensity_y - np.min(intensity_y)
    intensity_y = intensity_y / np.max(intensity_y)

    intensity_x_2 = np.sum(np_psf_t_x_2[:, :] ** 2, axis=1)
    intensity_x_2 = intensity_x_2 - np.min(intensity_x_2)
    intensity_x_2 = intensity_x_2 / np.max(intensity_x_2)
    intensity_y_2 = np.sum(np_psf_t_y_2[:, :] ** 2, axis=1)
    intensity_y_2 = intensity_y_2 - np.min(intensity_y_2)
    intensity_y_2 = intensity_y_2 / np.max(intensity_y_2)

    # Initial guess for the error function fit
    p0 = [0, 10]

    popt_x, _ = curve_fit(error_f, x_axis_psf_2, intensity_x_2, maxfev=8000, p0=p0)
    popt_y, _ = curve_fit(error_f, y_axis_psf_2, intensity_y_2, maxfev=8000, p0=p0)

    max_x = np.argmax(gaussian(x_axis_psf, *popt_x))
    max_y = np.argmax(gaussian(y_axis_psf, *popt_y))

    x0 = x_axis_psf[max_x]
    y0 = y_axis_psf[max_y]

    print("Center of the PSF: (" + str(x0) + ", " + str(y0) + ")")

    if plot:
        dydx = np.gradient(intensity_x, x_axis_psf)
        plt.plot(x_axis_psf, intensity_x, 'C0')
        plt.plot(x_axis_psf, error_f(x_axis_psf, *popt_x), 'C0--')
        plt.plot(x_axis_psf, dydx, 'C3')
        plt.plot(x_axis_psf, gaussian(x_axis_psf, *popt_x), 'C3--')
        plt.xlabel("y axis [mm]")
        plt.ylabel(r'$P/P_{\mathrm{max}}$')
        plt.title("PSF fit in x")
        plt.legend([r'Measured $P/P_{\mathrm{max}}$', r'Fit with $P_n(x)$', r'$\mathrm{d}P/\mathrm{d}x/P_{\mathrm{max}}$', r'Fit with $I_n(x)$'])
        plt.text(x_axis_psf[0], 0.1 * max(intensity_x), r"$w_x=$" + str(abs(round(popt_x[-1], 2))), fontsize=12)
        plt.show()

        dydx = np.gradient(intensity_y, y_axis_psf)
        plt.plot(y_axis_psf, intensity_y, 'C0')
        plt.plot(y_axis_psf, error_f(y_axis_psf, *popt_y), 'C0--')
        plt.plot(y_axis_psf, dydx, 'C3')
        plt.plot(y_axis_psf, gaussian(y_axis_psf, *popt_y), 'C3--')
        plt.xlabel("x axis [mm]")
        plt.ylabel(r'$P/P_{\mathrm{max}}$')
        plt.title("PSF fit in y")
        plt.legend([r'Measured $P/P_{\mathrm{max}}$', r'Fit with $P_n(y)$', r'$\mathrm{d}P/\mathrm{d}y/P_{\mathrm{max}}$', r'Fit with $I_n(y)$'])
        plt.text(y_axis_psf[0], 0.1 * max(intensity_y), r"$w_y=$" + str(abs(round(popt_y[-1], 2))), fontsize=12)
        plt.show()

        dx = 0.1
        dy = 0.1
        xx = np.arange(-4, 4 + dx, dx)
        yy = np.arange(-4, 4 + dy, dy)
        params_x = popt_x
        params_x[0] = 0.0
        params_y = popt_y
        params_y[0] = 0.0
        gauss_x = gaussian(xx, *params_x)
        gauss_x = gauss_x / np.max(gauss_x)
        gauss_y = gaussian(yy, *params_y)
        gauss_y = gauss_y / np.max(gauss_y)
        plt.figure()
        plt.plot(xx, gauss_x, label='PSF in x', color='C0')
        plt.plot(yy, gauss_y, label='PSF in y', color='C3')
        plt.xlabel("Position [mm]")
        plt.ylabel(r'$I/I_0$')
        plt.legend()
        plt.title("Beam profile")
        plt.show()

        create_psf_2d(gauss_x, gauss_y, xx, yy, plot=True)

    return x0, y0, popt_x, popt_y

# Create the filters for the frequency domains
def create_filters(n_filters, times, win_width, low_cut, high_cut, start_freq, end_freq, plot=False):
    """
    Create a set of bandpass filters with logarithmically spaced center frequencies.

    Parameters
    ----------
    n_filters : int
        Number of filters to create.
    times : array_like
        Time array corresponding to the signal.
    win_width : float
        Width of the Kaiser window.
    low_cut : float
        Lower cutoff frequency for the filters.
    high_cut : float
        Upper cutoff frequency for the filters.
    start_freq : float
        Starting frequency for the logarithmic spacing.
    end_freq : float
        Ending frequency for the logarithmic spacing.
    plot : bool, optional
        If True, plots the filter responses. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - filters : list of ndarray
            List of filter coefficients.
        - filt_freqs : list of float
            Center frequencies of the filters.
    """
    print(f"Creating {n_filters} log-spaced filters...")

    # Determine FIR length
    ntaps = len(times) // 5
    if ntaps % 2 == 0:
        ntaps += 1

    fs = len(times) / (times[-1] - times[0])

    # Logarithmic center frequencies
    filt_freqs = np.geomspace(start_freq, end_freq, num=n_filters)

    filters = []
    all_responses = []

    for i in range(n_filters):
        # Calculate lowcut and highcut
        if i == 0:
            lowcut = low_cut
        else:
            lowcut = np.sqrt(filt_freqs[i - 1] * filt_freqs[i])

        if i == n_filters - 1:
            highcut = high_cut
        else:
            highcut = np.sqrt(filt_freqs[i] * filt_freqs[i + 1])

        # Design filter
        taps = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=win_width)
        filters.append(taps)

        # Frequency response
        w, h = freqz(taps, 1, worN=1000, fs=fs)
        all_responses.append(np.abs(h))

        if plot:
            plt.plot(w, np.abs(h), label=f"{filt_freqs[i]:.2f} THz")

    if plot:
        # Sum of all filters
        total_response = np.sum(all_responses, axis=0)
        plt.plot(w, total_response, 'k--', linewidth=2, label="Sum of filters")

        plt.xlabel("Frequency [THz]")
        plt.ylabel("Gain")
        plt.title("Bandpass Filters and Total Response")
        plt.legend()
        plt.grid(True)
        plt.show()

    return filters, filt_freqs.tolist()

# Fit the beam widths for each window
def fit_beam_widths(x0, y0, x_axis_psf, y_axis_psf, np_psf_t_x, np_psf_t_y, filters, filt_freqs, w_max, nrange=None, plot=False):
    """
    Fit beam widths for a set of filters and compute their parameters.

    Parameters
    ----------
    x0 : float
        Initial guess for the x-axis center of the beam.
    y0 : float
        Initial guess for the y-axis center of the beam.
    x_axis_psf : array_like
        x-axis positions of the PSF.
    y_axis_psf : array_like
        y-axis positions of the PSF.
    np_psf_t_x : ndarray
        PSF traces along the x-axis.
    np_psf_t_y : ndarray
        PSF traces along the y-axis.
    filters : list of ndarray
        List of filters to apply to the PSF traces.
    filt_freqs : array_like
        Frequencies corresponding to the filters.
    w_max : float
        Maximum beam width for fitting.
    nrange : tuple of int, optional
        Range of indices to crop the PSF for fitting. Default is None.
    plot : bool, optional
        If True, plots the fitting results. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - xxx : ndarray
            x-axis positions for the Gaussian fit.
        - yyy : ndarray
            y-axis positions for the Gaussian fit.
        - popt_xs : ndarray
            Optimal parameters for the x-axis fits.
        - popt_ys : ndarray
            Optimal parameters for the y-axis fits.
        - popt_wx : ndarray
            Parameters for the beam width fit along the x-axis.
        - popt_wy : ndarray
            Parameters for the beam width fit along the y-axis.
    """
    w_xs = []
    w_ys = []
    x0s = []
    y0s = []
    popt_xs = []
    popt_ys = []

    dx = x_axis_psf[1] - x_axis_psf[0]
    dy = y_axis_psf[1] - y_axis_psf[0]

    # Cropping the PSF to improve the fit
    # as the right part of the measurement is noisy
    # and presents deviations from the error function
    if nrange is not None:
        n_min = nrange[0]
        n_max = nrange[1]
        x_axis_psf_2 = x_axis_psf[n_min:n_max]
        y_axis_psf_2 = y_axis_psf[n_min:n_max]
        np_psf_t_x_2 = np_psf_t_x[n_min:n_max,:]
        np_psf_t_y_2 = np_psf_t_y[n_min:n_max,:]
    else:
        x_axis_psf_2 = x_axis_psf
        y_axis_psf_2 = y_axis_psf
        np_psf_t_x_2 = np_psf_t_x
        np_psf_t_y_2 = np_psf_t_y

    # Initial guess for the error function fit
    popt_x = [x0, w_max]
    popt_y = [y0, w_max]

    # Bounds for the error function fit
    range_max = w_max * 1.5
    bounds_x = ([-range_max / 2, 0.01], [range_max / 2, w_max])
    bounds_y = ([-range_max / 2, 0.01], [range_max / 2, w_max])

    # Create the x and y axis for the Gaussian
    xxx = np.arange(-range_max, range_max + dx, dx)
    yyy = np.arange(-range_max, range_max + dy, dy)

    print("Fitting beam widths...")
    for nf in tqdm(range(len(filters))):
        # x axis
        np_psf_t_x_filtered = np.zeros(np_psf_t_x_2.shape)
        for nx in range(np_psf_t_x_2.shape[0]):
            # Filter the signal with the window
            filtered = signal.convolve(np_psf_t_x_2[nx,:], filters[nf], mode='same')
            np_psf_t_x_filtered[nx,:] = filtered

        # Compute the intensity of the filtered signal
        intensity_x = np.sum(np_psf_t_x_filtered[:,:] ** 2, axis=1)
        intensity_x = intensity_x - np.min(intensity_x)
        intensity_x = intensity_x / np.max(intensity_x)
        # Fit the error function to the intensity
        popt_x, _ = curve_fit(error_f, x_axis_psf_2, intensity_x, maxfev=8000, p0=popt_x, bounds=bounds_x)
        x_offset, w_max = popt_x
        bounds_x = ([-w_max / 2 + x_offset, 0.0], [w_max / 2 + x_offset, w_max])
        w_xs.append(abs(popt_x[-1]))
        x0s.append(popt_x[0])
        # Save the parameters
        popt_xs.append(popt_x)

        # y axis
        np_psf_t_y_filtered = np.zeros(np_psf_t_y_2.shape)
        for ny in range(np_psf_t_y_2.shape[0]):
            # Filter the signal with the window
            filtered = signal.convolve(np_psf_t_y_2[ny,:], filters[nf], mode='same')
            np_psf_t_y_filtered[ny,:] = filtered

        # Compute the intensity of the filtered signal
        intensity_y = np.sum(np_psf_t_y_filtered[:,:] ** 2, axis=1)
        intensity_y = intensity_y - np.min(intensity_y)
        intensity_y = intensity_y / np.max(intensity_y)
        # Fit the error function to the intensity
        popt_y, _ = curve_fit(error_f, y_axis_psf_2, intensity_y, maxfev=8000, p0=popt_y, bounds=bounds_y)
        y_offset, w_max = popt_y
        bounds_y = ([-w_max / 2 + y_offset, 0.0], [w_max / 2 + y_offset, w_max])
        w_ys.append(abs(popt_y[-1]))
        y0s.append(popt_y[0])
        popt_ys.append(popt_y)

    n_min = 0
    popt_wx, _ = curve_fit(beam_w, filt_freqs[n_min:], w_xs[n_min:], maxfev=8000)
    popt_wy, _ = curve_fit(beam_w, filt_freqs[n_min:], w_ys[n_min:], maxfev=8000)

    if plot:

        plt.plot(filt_freqs, w_xs, 'C0')
        plt.plot(filt_freqs, w_ys, 'C3')
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Beam width [mm]")
        plt.title("Beam width as a function of frequency")
        plt.legend(["Beam width in x", "Beam width in y"])
        plt.show()

        plt.plot(filt_freqs, x0s, 'C0')
        plt.plot(filt_freqs, y0s, 'C3')
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Position of the center [mm]")
        plt.title("Center of the PSF as a function of frequency")
        plt.legend(["Center of the PSF in x", "Center of the PSF in y"])
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Frequency [THz]")
        ax1.set_ylabel("Beam width [mm]")
        p1, = ax1.plot(filt_freqs, w_xs, 'C0', label="Beam width in x")
        p2, = ax1.plot(filt_freqs, w_ys, 'C3', label="Beam width in y")
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.set_ylabel("Position of the center [mm]")  # we already handled the x-label with ax1
        p3, = ax2.plot(filt_freqs, x0s, 'C0--', label="Position of the center in x")
        p4, = ax2.plot(filt_freqs, y0s, 'C3--', label="Position of the center in y")
        ax1.legend(handles=[p1, p2, p3, p4])
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Width and center of the PSF as a function of frequency")
        plt.show()

        plt.plot(filt_freqs, w_xs, 'C0')
        plt.plot(filt_freqs, w_ys, 'C3')
        plt.plot(filt_freqs, beam_w(filt_freqs, *popt_wx), 'C0--')
        plt.plot(filt_freqs, beam_w(filt_freqs, *popt_wy), 'C3--')
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Beam width [mm]")
        plt.title("Beam width as a function of frequency: fit")
        plt.legend(["Beam width in x", "Beam width in y", "Fit in x", "Fit in y"])
        plt.show()

        plt.plot(x0s, y0s, 'C0')
        plt.xlabel("Center of the PSF in x [mm]")
        plt.ylabel("Center of the PSF in y [mm]")
        plt.title("Position of the center [mm]")
        plt.show()

    return xxx, yyy, np.array(popt_xs), np.array(popt_ys), popt_wx, popt_wy

def range_max_min(range_max, wmin):
    """
    Ensure the range maximum is not less than a specified minimum.

    Parameters
    ----------
    range_max : float
        The current maximum range value.
    wmin : float
        The minimum allowable range value.

    Returns
    -------
    float
        Adjusted range maximum.
    """
    if range_max < wmin:
        range_max = wmin
    return range_max

# Richardson-Lucy deconvolution working in the frequency domain
# Computes the deconvolution for one window
def richardson_lucy_worker(nf):
    """
    Perform Richardson-Lucy deconvolution for a specific filter index.

    Parameters
    ----------
    nf : int
        Index of the filter to be applied.

    Returns
    -------
    tuple
        A tuple containing:
        - traces_filtered : ndarray
            Deconvolved and filtered traces reshaped to the original dimensions.
        - nf : int
            The filter index used for processing.
    """
    shape = traces_glob.shape
    traces_flatten = traces_glob.reshape(shape[0] * shape[1], shape[2])
    traces_filtered = np.zeros(traces_flatten.shape)
    for nn in range(traces_flatten.shape[0]):
        traces_filtered[nn,:] = signal.convolve(traces_flatten[nn], filters_glob[nf], mode='same')
    traces_filtered = traces_filtered.reshape(shape[0], shape[1], shape[-1])
    range_max_x = (popt_xs_glob[nf][1] + np.abs(popt_xs_glob[nf][0])) * 3
    range_max_y = (popt_ys_glob[nf][1] + np.abs(popt_ys_glob[nf][0])) * 3
    range_max_x = range_max_min(range_max_x, 2.5)
    range_max_y = range_max_min(range_max_y, 2.5)
    dx = x_axis_psf_glob[1] - x_axis_psf_glob[0]
    dy = y_axis_psf_glob[1] - y_axis_psf_glob[0]
    range_max_x = (range_max_x // dx) * dx + dx
    range_max_y = (range_max_y // dy) * dy + dy
    x = np.arange(-range_max_x, range_max_x + dx, dx)
    y = np.arange(-range_max_y, range_max_y + dy, dy)
    params = popt_xs_glob[nf]
    gaussian_x = gaussian(x, *params)
    params = popt_ys_glob[nf]
    gaussian_y = gaussian(y, *params)
    if type_glob == 'transmission':
        _, _, psf_2d = create_psf_2d(gaussian_x, gaussian_y, x, y, plot=False)
    elif type_glob == 'reflectance':
        _, _, psf_2d = create_psf_2d(gaussian_x, np.flip(gaussian_y), x, y, plot=False)
    else:
        raise ValueError("The scan type must be either 'transmission' or 'reflectance'")
    image_filtered = np.sum(traces_filtered**2, axis=2) + 1.0 # Avoid division by zero
    w_min = np.min(np.array(popt_xs_glob)[:,1])
    w_max = np.max(np.array(popt_xs_glob)[:,1])
    num_iter_min = 1
    num_iter = (popt_xs_glob[nf][1] - w_min) / (w_max - w_min) * (num_iter_glob - num_iter_min) + num_iter_min
    num_iter = int(num_iter)
    deconvolved_filtered = richardson_lucy_unclipped(image_filtered, psf_2d, num_iter=num_iter)
    #deconvolved_filtered = restoration.richardson_lucy(image_filtered, psf_2d, num_iter=num_iter, clip=False)

    # Compute gains
    deconvolution_gains = deconvolved_filtered / image_filtered
    deconvolution_gains = np.sqrt(deconvolution_gains)
    deconvolution_gains_flatten = deconvolution_gains.reshape(shape[0] * shape[1])
    traces_filtered = traces_filtered.reshape(shape[0] * shape[1], shape[2])
    # Apply the gains to the filtered traces
    for nn in range(traces_filtered.shape[0]):
        traces_filtered[nn,:] = traces_filtered[nn,:] * deconvolution_gains_flatten[nn]
    return traces_filtered.reshape(shape), nf


# Richardson-Lucy deconvolution working in the frequency domain
# Computes the deconvolution for all filters
# This function uses multiprocessing to speed up the computation
def richardson_lucy_freq(traces, x_axis_psf, y_axis_psf, popt_xs, popt_ys, filters, filt_freqs, num_iter, scan_type, center_cor=True, multithread=False):
    """
    Perform Richardson-Lucy deconvolution across multiple filters.

    Parameters
    ----------
    traces : ndarray
        Input traces to be deconvolved.
    x_axis_psf : array_like
        x-axis positions of the PSF.
    y_axis_psf : array_like
        y-axis positions of the PSF.
    popt_xs : ndarray
        Optimal parameters for the x-axis PSF fits.
    popt_ys : ndarray
        Optimal parameters for the y-axis PSF fits.
    filters : list of ndarray
        List of filters to apply to the traces.
    filt_freqs : array_like
        Frequencies corresponding to the filters.
    num_iter : int
        Number of iterations for the deconvolution process.
    scan_type : {'transmission', 'reflectance'}
        Type of scan being processed.
    center_cor : bool, optional
        If True, applies center correction to the PSF parameters. Default is True.
    multithread : bool, optional
        If True, enables multithreaded processing. Default is False.

    Returns
    -------
    ndarray
        Deconvolved traces summed across all filters.
    """
    # This is necessary to avoid large overhead when using multiprocessing
    # It's dirty but it works
    global traces_glob, x_axis_psf_glob, y_axis_psf_glob, popt_xs_glob, popt_ys_glob, filters_glob, num_iter_glob, filt_freqs_glob, type_glob, center_cor_glob
    deconvolved = np.zeros(traces.shape)
    traces_glob = traces
    x_axis_psf_glob = x_axis_psf
    y_axis_psf_glob = y_axis_psf
    popt_xs_glob = np.array(popt_xs)
    popt_ys_glob = np.array(popt_ys)
    filt_freqs_glob = filt_freqs
    filters_glob = filters
    num_iter_glob = num_iter
    type_glob = scan_type
    center_cor_glob = center_cor
    if not center_cor:
        popt_xs_glob[:,0] = 0.0
        popt_ys_glob[:,0] = 0.0
    n_filters = len(filters)
    if multithread:
        with tqdm(total=n_filters) as pbar:
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = [executor.submit(richardson_lucy_worker, nf) for nf in np.arange(n_filters)]
                for future in as_completed(futures):
                    pbar.update(1)
                    traces, nf = future.result()
                    deconvolved += traces
    else:
        for nf in tqdm(range(n_filters)):
            traces, nf = richardson_lucy_worker(nf)
            deconvolved += traces
    return deconvolved


# Richardson-Lucy deconvolution working in the frequency domain
# Computes the deconvolution for one window
def wiener_worker(nf):
    """
    Perform Wiener deconvolution for a specific filter index.

    Parameters
    ----------
    nf : int
        Index of the filter to be applied.

    Returns
    -------
    ndarray
        Deconvolved and filtered traces reshaped to the original dimensions.
    """
    shape = traces_glob.shape
    traces_flatten = traces_glob.reshape(shape[0] * shape[1], shape[2])
    traces_filtered = np.zeros(traces_flatten.shape)
    for nn in range(traces_flatten.shape[0]):
        traces_filtered[nn,:] = signal.convolve(traces_flatten[nn], filters_glob[nf], mode='same')
    traces_filtered = traces_filtered.reshape(shape[0], shape[1], shape[-1])
    range_max_x = (popt_xs_glob[nf][1] + np.abs(popt_xs_glob[nf][0])) * 3
    range_max_y = (popt_ys_glob[nf][1] + np.abs(popt_ys_glob[nf][0])) * 3
    range_max_x = range_max_min(range_max_x, 2.5)
    range_max_y = range_max_min(range_max_y, 2.5)
    dx = x_axis_psf_glob[1] - x_axis_psf_glob[0]
    dy = y_axis_psf_glob[1] - y_axis_psf_glob[0]
    range_max_x = (range_max_x // dx) * dx + dx
    range_max_y = (range_max_y // dy) * dy + dy
    x = np.arange(-range_max_x, range_max_x + dx, dx)
    y = np.arange(-range_max_y, range_max_y + dy, dy)
    params = popt_xs_glob[nf]
    gaussian_x = gaussian(x, *params)
    params = popt_ys_glob[nf]
    gaussian_y = gaussian(y, *params)
    if type_glob == 'transmission':
        _, _, psf_2d = create_psf_2d(gaussian_x, gaussian_y, x, y, plot=False)
    elif type_glob == 'reflectance':
        _, _, psf_2d = create_psf_2d(gaussian_x, np.flip(gaussian_y), x, y, plot=False)
    else:
        raise ValueError("The scan type must be either 'transmission' or 'reflectance'")

    image_filtered = np.sum(traces_filtered**2, axis=2) + 1.0 # Avoid division by zero
    pad = int(len(gaussian_x) // 4)
    image_filtered = np.pad(image_filtered, pad, 'minimum')
    #deconvolved_filtered, _ = restoration.unsupervised_wiener(image_filtered, psf_2d, clip = False)
    deconvolved_filtered = restoration.wiener(image_filtered, psf_2d, balance=balance_glob, clip = False)
    deconvolved_filtered = deconvolved_filtered[pad:-pad, pad:-pad]
    deconvolved_filtered[deconvolved_filtered < 0] = 0

    # Compute gains
    deconvolution_gains = deconvolved_filtered / image_filtered[pad:-pad, pad:-pad]
    deconvolution_gains = np.sqrt(deconvolution_gains)
    deconvolution_gains_flatten = deconvolution_gains.reshape(shape[0] * shape[1])
    traces_filtered = traces_filtered.reshape(shape[0] * shape[1], shape[2])
    # Apply the gains to the filtered traces
    for nn in range(traces_filtered.shape[0]):
        traces_filtered[nn,:] = traces_filtered[nn,:] * deconvolution_gains_flatten[nn]
    return traces_filtered.reshape(shape)


# Richardson-Lucy deconvolution working in the frequency domain
# Computes the deconvolution for all filters
# This function uses multiprocessing to speed up the computation
def wiener_freq(traces, x_axis_psf, y_axis_psf, popt_xs, popt_ys, filters, filt_freqs, scan_type, balance, center_cor=True, multithread=False):
    """
    Perform Wiener deconvolution across multiple filters.

    Parameters
    ----------
    traces : ndarray
        Input traces to be deconvolved.
    x_axis_psf : array_like
        x-axis positions of the PSF.
    y_axis_psf : array_like
        y-axis positions of the PSF.
    popt_xs : ndarray
        Optimal parameters for the x-axis PSF fits.
    popt_ys : ndarray
        Optimal parameters for the y-axis PSF fits.
    filters : list of ndarray
        List of filters to apply to the traces.
    filt_freqs : array_like
        Frequencies corresponding to the filters.
    scan_type : {'transmission', 'reflectance'}
        Type of scan being processed.
    balance : float
        Balance parameter for the Wiener deconvolution.
    center_cor : bool, optional
        If True, applies center correction to the PSF parameters. Default is True.
    multithread : bool, optional
        If True, enables multithreaded processing. Default is False.

    Returns
    -------
    ndarray
        Deconvolved traces summed across all filters.
    """
    # This is necessary to avoid large overhead when using multiprocessing
    # It's dirty but it works
    global traces_glob, x_axis_psf_glob, y_axis_psf_glob, popt_xs_glob, popt_ys_glob, filters_glob, filt_freqs_glob, type_glob, center_cor_glob, balance_glob
    deconvolved = np.zeros(traces.shape)
    traces_glob = traces
    x_axis_psf_glob = x_axis_psf
    y_axis_psf_glob = y_axis_psf
    popt_xs_glob = np.array(popt_xs)
    popt_ys_glob = np.array(popt_ys)
    filt_freqs_glob = filt_freqs
    filters_glob = filters
    type_glob = scan_type
    center_cor_glob = center_cor
    balance_glob = balance
    n_filters = len(filters)
    if not center_cor:
        popt_xs_glob[:,0] = 0.0
        popt_ys_glob[:,0] = 0.0
    if multithread:
        with tqdm(total=n_filters) as pbar:
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = [executor.submit(wiener_worker, nf) for nf in np.arange(n_filters)]
                for future in as_completed(futures):
                    pbar.update(1)
                    deconvolved += future.result()
    else:
        for nf in tqdm(range(n_filters)):
            deconvolved += wiener_worker(nf)
    return deconvolved
