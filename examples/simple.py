from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile

from thz_deconvolution.deconvolution import Deconvolution

if __name__ == "__main__":
    DeconvolutionFilter = Deconvolution(
        knife_edge_x_path=Path("../psf_data/example_beam_width/measurement_x/1750085285.8557956_data.thz"),
        knife_edge_y_path=Path("../psf_data/example_beam_width/measurement_y/1750163177.929295_data.thz"),
        low_cut=0.5,
        high_cut=5.0,
    )

    # Decon = Deconvolution(psf_path=Path("../psf_data/example_beam_width/psf.npz"))

    with DotthzFile(Path("../sample_data/resolution_target_sample.thzimg")) as psf_data:
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
    plt.title("Raw scan (summed pulse energy)")
    plt.xlabel("y index")
    plt.ylabel("x index")
    plt.colorbar(label="Summed intensity [a.u.]")
    plt.show()

    deconvolved_traces = DeconvolutionFilter.apply_deconvolution(traces, x_min, x_max, nx, y_min, y_max, ny,
                                                                 max_iter=100)

    plt.imshow(np.sum(deconvolved_traces ** 2, axis=2), cmap='gray')
    plt.title("Deconvolved scan (summed pulse energy)")
    plt.xlabel("y index")
    plt.ylabel("x index")
    plt.colorbar(label="Summed intensity [a.u.]")
    plt.show()

    raw_pulse = traces[nx // 2, ny // 2, :]
    deconvolved_trace = DeconvolutionFilter.apply_single_pulse_deconvolution(raw_pulse, max_iter=100)

    plt.plot(times, raw_pulse)
    plt.plot(times, deconvolved_trace)
    plt.title("Single-pulse deconvolution")
    plt.xlabel("Time [ps]")
    plt.ylabel("Amplitude [a.u.]")
    plt.legend(["Raw pulse", "Deconvolved pulse"])
    plt.show()
