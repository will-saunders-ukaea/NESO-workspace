import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.ticker as mtick
import math

plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True)
plt.rc("savefig", dpi=500)
plt.rc("xtick", labelsize="small")
plt.rc("legend", fontsize="medium")
plt.rc("legend", edgecolor="lightgray")
plt.rc("ytick", labelsize="small")
plt.rcParams.update({"font.size": 20})

plt.rc("font", family="serif", serif="Times New Roman")
params = {"text.latex.preamble": r"\usepackage{amsmath}"}

plt.rcParams.update(params)

import h5py

if __name__ == "__main__":
    
    assert len(sys.argv) > 1
    h5file = h5py.File(sys.argv[1], 'r')
    flops = np.array(h5file["flops_evaluation"]) * 1E-9
    size = h5file.attrs["mpi_size"]
    num_modes = h5file.attrs["num_modes"]
    theoretical_max = float(sys.argv[2]) if len(sys.argv) > 2 else None

    x_positions = np.zeros((size, 6))
    
    bar_width = 0.2
    bar_stride = 0.2 * 7


    for rx in range(size):
        for gx in range(6):
            x_positions[rx, gx] = rx * bar_stride + gx * bar_width
  
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    labels = [
        "Quadrilateral",
        "Triangle",
        "Hexahedron",
        "Prism",
        "Tetrahedron",
        "Pyramid",
    ]

    for gx in range(6):
        ax.bar(x_positions[:, gx], flops[:, gx], width=bar_width, label=labels[gx])
    plt.xticks([(r + 0.5) * bar_stride for r in range(size)], [str(rx) for rx in range(size)])

    
    if theoretical_max is None:
        ax_max = 1.02 * flops.max();
    else:
        ax_max = 1.02 * theoretical_max
        ax.axhline(y=theoretical_max, color='k', linestyle='--', linewidth=2, label='Rpeak')
    ax.set_ylim(0, ax_max)

    ax.legend()
    ax.set_ylabel(r"GFLOPs")
    ax.set_xlabel(r"MPI rank")
    fig.savefig("benchmark_flops_evaluation.pdf", bbox_inches="tight")







