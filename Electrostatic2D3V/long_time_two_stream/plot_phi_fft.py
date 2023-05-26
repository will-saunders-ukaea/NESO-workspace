import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.ticker as mtick
import h5py
import math


def get_metadata(data):
    """
    Get the number of evaluation points per sample and number of samples.
    """
    step = list(data.keys())[0]
    step_data = data[step]
    num_points = len(step_data["x"])
    num_steps = len(data.keys())
    return num_points, num_steps

def get_step_index(step_name):
    """
    Get the step index from the step name.
    """
    return int(step_name.split("#")[-1])

def get_step_keys(data):
    """
    Returns sorted list of step names.
    """
    return sorted(data.keys(), key=get_step_index)

def get_data(data):
    """
    Get the sample data into a numpy array
    """

    num_points, num_steps = get_metadata(data)
    step_keys = get_step_keys(data)
    
    samples = np.zeros((num_steps, num_points))

    for step_index, step_name in enumerate(step_keys):
        index_0 = np.array(data[step_name]["INDEX_0"])
        values = np.array(data[step_name]["FIELD_EVALUATION_0"])
        for ix, xx in enumerate(index_0):
            value = values[ix]
            samples[step_index, xx] = value
    
    return samples


def get_fft(data):
    """
    Take fft of sample points.
    """
    fft_data = np.fft.rfft(data)
    #test = np.zeros_like(fft_data)
    #nrow = fft_data.shape[0]
    #for rowx in range(nrow):
    #    test[rowx,:] = np.fft.rfft(data[rowx, :])
    #print(np.linalg.norm(test.ravel() - fft_data.ravel()))

    return fft_data


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
"""
Run like:
    python3 plot_phi_fft.py Electrostatic2D3V_line_field_evaluations.h5part
""")
        quit()

    data_h5 = h5py.File(sys.argv[1], "r")
    num_points, num_steps = get_metadata(data_h5)

    data = get_data(data_h5)
    fft_data = get_fft(data)

    abs_fft_data = np.abs(fft_data)

    freq = np.fft.rfftfreq(num_points)
    ax = plt.figure().add_subplot(projection='3d')
    
    freq_cutoff = 10
    for line in range(num_steps):
        ax.plot(freq[:freq_cutoff], abs_fft_data[line, :freq_cutoff], line)

    ax.set_xlabel("Frequency")
    ax.set_zlabel("Time sample")
    ax.set_ylabel("Amplitude")

    plt.show()



