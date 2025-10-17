from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from pydotthz import DotthzFile
from thz_deconvolution import richardson_lucy_freq, load_knife_edge_meas, fit_mean_beam, fit_beam_widths, create_psf_2d, \
    create_filters, gaussian
import numpy as np


class Deconvolution:

    def __init__(self, knife_edge_x_path: Optional[Path] = None, knife_edge_y_path: Optional[Path] = None,
                 psf_path: Optional[Path] = None,
                 low_cut=0.1,
                 high_cut=10.0,
                 start_freq=0.25,
                 end_freq=4.0,
                 win_width=0.5,
                 n_filters=20,
                 w_max=30,
                 show_plot=False):

        if knife_edge_x_path is None and knife_edge_y_path is None:
            if psf_path is None:
                raise ValueError("At least a psf or the knife edge measurement files must be provided.")
            else:

                data = np.load(psf_path)

                low_cut = data['low_cut']
                high_cut = data['high_cut']
                start_freq = data['start_freq']
                end_freq = data['end_freq']
                n_filters = data['n_filters']
                self.filters = data['filters']
                self.filt_freqs = data['filt_freqs']
                self.popt_xs = data['[x_0, w_x]']
                self.popt_ys = data['[y_0, w_y]']
        else:

            x_psf_lr, y_psf_lr, np_psf_t_x_lr, np_psf_t_y_lr, times_psf = load_knife_edge_meas(knife_edge_x_path,
                                                                                               knife_edge_y_path)

            x_psf_lr = np.split(x_psf_lr, 2)
            y_psf_lr = np.split(y_psf_lr, 2)
            np_psf_t_x_lr = np.split(np_psf_t_x_lr, 2)
            np_psf_t_y_lr = np.split(np_psf_t_y_lr, 2)

            # Flipping the left part
            x_psf_lr[0] = -np.flip(x_psf_lr[0])
            y_psf_lr[0] = -np.flip(y_psf_lr[0])
            np_psf_t_x_lr[0] = np.flip(np_psf_t_x_lr[0])
            np_psf_t_y_lr[0] = np.flip(np_psf_t_y_lr[0])

            popt_xs_lr = []
            popt_ys_lr = []

            for x_psf, y_psf, np_psf_t_x, np_psf_t_y in zip(x_psf_lr, y_psf_lr, np_psf_t_x_lr, np_psf_t_y_lr):
                x_psf -= np.mean(x_psf)
                y_psf -= np.mean(y_psf)

                n_min = 0
                n_max = -1
                x0, y0, popt_x, popt_y = fit_mean_beam(
                    x_psf, y_psf, np_psf_t_x, np_psf_t_y, [n_min, n_max], plot=show_plot)

                # Create the PSF
                x_start = np.abs(x_psf[0])
                y_start = np.abs(y_psf[0])
                dx = np.abs(x_psf[1] - x_psf[0])
                dy = np.abs(y_psf[1] - y_psf[0])
                xx = np.arange(-x_start, x_start + dx, dx)
                yy = np.arange(-y_start, y_start + dy, dy)

                gauss_x = gaussian(xx, 0.0, popt_x[1])
                gauss_y = gaussian(yy, 0.0, popt_y[1])
                gauss_x = gauss_x / np.max(gauss_x)
                gauss_y = gauss_y / np.max(gauss_y)

                _, _, psf_2d = create_psf_2d(gauss_x, gauss_y, xx, yy, plot=show_plot)

                self.filters, self.filt_freqs = create_filters(
                    n_filters, times_psf, win_width, low_cut, high_cut, start_freq, end_freq, plot=show_plot)

                n_min = 0
                n_max = -1
                _, _, popt_xs, popt_ys, _, _ = fit_beam_widths(
                    x0, y0, x_psf, y_psf, np_psf_t_x, np_psf_t_y, self.filters, self.filt_freqs, w_max, [n_min, n_max],
                    plot=show_plot)

                popt_xs_lr.append(popt_xs)
                popt_ys_lr.append(popt_ys)

            popt_xs_lr = np.array(popt_xs_lr)
            popt_ys_lr = np.array(popt_ys_lr)

            popt_xs_lr[0].T[0] = -popt_xs_lr[0].T[0]
            popt_ys_lr[0].T[0] = -popt_ys_lr[0].T[0]

            # Averaging
            self.popt_xs = (popt_xs_lr[0] + popt_xs_lr[1]) / 2
            self.popt_ys = (popt_ys_lr[0] + popt_ys_lr[1]) / 2

            self.popt_xs.T[0] -= np.mean(self.popt_xs.T[0])
            self.popt_ys.T[0] -= np.mean(self.popt_ys.T[0])

            # Save the data to a .npz file
            data = {
                'low_cut': low_cut,  # float: low cut-off frequency
                'high_cut': high_cut,  # float: high cut-off frequency
                'start_freq': start_freq,  # float: start frequency for filters
                'end_freq': end_freq,  # float: end frequency for filters
                'n_filters': n_filters,  # int: number of filters
                # ndarray: filter coefficients, shape (n_filters, len(times_psf) // 5)
                'filters': self.filters,
                # ndarray: filter frequencies, shape (n_filters,)
                'filt_freqs': self.filt_freqs,
                # ndarray: fitted x parameters, shape (n_filters, 2)
                '[x_0, w_x]': self.popt_xs,
                '[y_0, w_y]': self.popt_ys  # ndarray: fitted y parameters, shape (n_filters, 2)
            }

            np.savez(Path("psf_data/psf_new.npz"), **data)

    def apply_deconvolution(self, scan, x_min, x_max, nx, y_min, y_max, ny, max_iter=500):
        # Initialize the maximum number of iterations for Richardson-Lucy deconvolution
        meas_type = 'reflectance'  # or meas_type == 'transmission', to mirror the PSF
        # Perform Richardson-Lucy deconvolution in the frequency domain

        xx = np.linspace(x_min, x_max, nx)
        yy = np.linspace(y_min, y_max, ny)

        deconvolved_traces = richardson_lucy_freq(scan, xx, yy, self.popt_xs, self.popt_ys, self.filters,
                                                  self.filt_freqs, max_iter,
                                                  scan_type=meas_type)
        return deconvolved_traces

if __name__ == "__main__":
    Decon = Deconvolution(
        knife_edge_x_path=Path("psf_data/example_beam_width/measurement_x/data/1750085285.8557956_data.thz"),
        knife_edge_y_path=Path("psf_data/example_beam_width/measurement_y/data/1750163177.929295_data.thz")
    )

    # Decon = Deconvolution(psf_path=Path("psf_data/psf.npz"))

    with DotthzFile(Path("sample_data/resolution_target_sample.thzimg")) as psf_data:
        key = list(psf_data.keys())[0]
        dx = float(psf_data[key].metadata['dx [mm]'])
        dy = float(psf_data[key].metadata['dy [mm]'])
        x_min = float(psf_data[key].metadata['x_min [mm]'])
        x_max = float(psf_data[key].metadata['x_max [mm]'])
        y_min = float(psf_data[key].metadata['y_min [mm]'])
        y_max = float(psf_data[key].metadata['y_max [mm]'])
        nx = int(float(psf_data[key].metadata['width']))
        ny = int(float(psf_data[key].metadata['height']))
        datasets = psf_data[key].datasets

        # from the first dataset, extract the image:
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    plt.imshow(np.sum(traces ** 2, axis=2), cmap='gray')
    plt.show()

    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, nx, y_min, y_max, ny, max_iter=100)

    plt.imshow(np.sum(deconvolved_traces ** 2, axis=2), cmap='gray')
    plt.show()
