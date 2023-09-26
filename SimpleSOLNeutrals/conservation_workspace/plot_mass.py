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

import pandas as pd

def plot_mass():

    df = pd.read_csv("mass_recording.csv")
    print(df)
    dt = 0.005

    times = [sx * dt for sx in df["step"]]


    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax2.plot(times, df["relative_error"], color="k", label="Relative Error", linestyle="--")
    ax.plot(times, df["mass_particles"], color="r", label="Particle Mass", linewidth=3)
    ax.plot(times, df["mass_fluid"], color="b", label="Fluid Mass", linewidth=3, linestyle="-.")

    ax2.set_ylim([1.0e-17, 1.0e-14])

    ax.legend()

    ax.set_ylabel(r"Mass")
    ax2.set_ylabel(r"Relative Error")
    ax.set_xlabel(r"Time")
    fig.savefig("system_mass.pdf", bbox_inches="tight")



if __name__ == "__main__":
    
    plot_mass()
