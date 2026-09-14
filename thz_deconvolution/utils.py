from scipy.signal.windows import tukey
from scipy.signal import convolve
from skimage import restoration
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
from scipy.signal import firwin, kaiser_atten, kaiser_beta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import signal
import multiprocess as mp
import numpy as np

from .psf import create_psf_2d, gaussian

__all__ = [
    "gaussian",
    "create_psf_2d",
    "richardson_lucy",
    "richardson_lucy_unclipped",
    "blackman_func",
    "toptica_window",
    "zero_padding",
    "get_fft",
    "get_fft_c",
    "extract_substring",
    "bandpass_kaiser",
    "zero_pad",
    "get_windowed_signal",
    "range_max_min",
    "wiener_worker",
    "wiener_freq",
]


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
    psf_T = np.flip(psf)  # Flipped PSF
    u = d.copy()  # Initial guess
    eps = 1e-12  # Regularization parameter to avoid division by zero
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
    d = np.pad(d, pad, 'reflect') if pad > 0 else d
    psf = psf / np.sum(psf)
    psf_T = np.flip(psf)  # Flipped PSF
    u = d.copy()  # Initial guess
    eps = 1e-12  # Regularization parameter to avoid division by zero
    for _ in range(num_iter):
        u = np.multiply(u, convolve(d / (convolve(u, psf, mode='same') + eps), psf_T, mode='same'))
    # `-pad` would wrap to an empty slice when pad == 0, so slice explicitly.
    return u[pad:u.shape[0] - pad, pad:u.shape[1] - pad] if pad > 0 else u


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
        return t, p, f, a, np.abs(arg)
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


# Kaiser windowed FIR filter
def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    atten = kaiser_atten(ntaps, width / (0.5 * fs))
    beta = kaiser_beta(atten)
    if lowcut <= 0.0:
        cutoffs = highcut
        pass_zero = 'lowpass'
    elif highcut >= 0.5 * fs:
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
        traces_filtered[nn, :] = signal.convolve(traces_flatten[nn], filters_glob[nf], mode='same')
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
        _, _, psf_2d = create_psf_2d(gaussian_x, gaussian_y, x, y, dx, dy)
    elif type_glob == 'reflectance':
        _, _, psf_2d = create_psf_2d(gaussian_x, np.flip(gaussian_y), x, y, dx, dy)
    else:
        raise ValueError("The scan type must be either 'transmission' or 'reflectance'")

    image_filtered = np.sum(traces_filtered ** 2, axis=2) + 1.0  # Avoid division by zero
    pad = int(len(gaussian_x) // 4)
    image_filtered = np.pad(image_filtered, pad, 'minimum')
    # deconvolved_filtered, _ = restoration.unsupervised_wiener(image_filtered, psf_2d, clip = False)
    deconvolved_filtered = restoration.wiener(image_filtered, psf_2d, balance=balance_glob, clip=False)
    deconvolved_filtered = deconvolved_filtered[pad:-pad, pad:-pad]
    deconvolved_filtered[deconvolved_filtered < 0] = 0

    # Compute gains
    deconvolution_gains = deconvolved_filtered / image_filtered[pad:-pad, pad:-pad]
    deconvolution_gains = np.sqrt(deconvolution_gains)
    deconvolution_gains_flatten = deconvolution_gains.reshape(shape[0] * shape[1])
    traces_filtered = traces_filtered.reshape(shape[0] * shape[1], shape[2])
    # Apply the gains to the filtered traces
    for nn in range(traces_filtered.shape[0]):
        traces_filtered[nn, :] = traces_filtered[nn, :] * deconvolution_gains_flatten[nn]
    return traces_filtered.reshape(shape)


# Richardson-Lucy deconvolution working in the frequency domain
# Computes the deconvolution for all filters
# This function uses multiprocessing to speed up the computation
def wiener_freq(traces, x_axis_psf, y_axis_psf, popt_xs, popt_ys, filters, filt_freqs, scan_type, balance,
                center_cor=True, multithread=False):
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
        popt_xs_glob[:, 0] = 0.0
        popt_ys_glob[:, 0] = 0.0
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
