import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import h5py
import xml.etree.ElementTree as ET

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


if __name__ == "__main__":

    if (len(sys.argv) < 5) or ("--help" in sys.argv) or ("-h" in sys.argv):
        print(
            """
Plots energies from electrostatic PIC. Call with:

python3 plot_energy.py input.xml field_energy.h5 potential_energy.h5 kinetic_energy.h5

"""
        )
        quit()

    xml_tree = ET.parse(sys.argv[1])
    xml_root = xml_tree.getroot()

    parameters = xml_root.find("CONDITIONS").find("PARAMETERS").findall("P")

    dt = -1
    for px in parameters:
        text = [tx for tx in px.itertext()][0].strip()
        if text.startswith("particle_time_step"):
            dt = float(text.split("=")[-1])
    assert dt > 0.0

    field_energy = h5py.File(sys.argv[2], "r")
    keys = sorted(field_energy.keys(), key=lambda x: int(x.split("#")[-1]))
    N = len(keys)
    x = np.zeros((N,))
    y = np.zeros((N,))
    for keyi, keyx in enumerate(keys):
        x[keyi] = dt * int(keyx.split("#")[-1])
        y[keyi] = field_energy[keyx]["field_energy"][0]

    potential_energy = h5py.File(sys.argv[3], "r")
    potential_keys = sorted(
        potential_energy.keys(), key=lambda x: int(x.split("#")[-1])
    )
    N_potential = len(potential_keys)
    potential_x = np.zeros((N_potential,))
    potential_y = np.zeros((N_potential,))
    for keyi, keyx in enumerate(potential_keys):
        potential_x[keyi] = dt * int(keyx.split("#")[-1])
        potential_y[keyi] = 0.5 * potential_energy[keyx]["potential_energy"][0]

    kinetic_energy = h5py.File(sys.argv[4], "r")
    kinetic_keys = sorted(
        kinetic_energy.keys(), key=lambda x: int(x.split("#")[-1])
    )
    N_kinetic = len(kinetic_keys)
    kinetic_x = np.zeros((N_kinetic,))
    kinetic_y = np.zeros((N_kinetic,))
    for keyi, keyx in enumerate(kinetic_keys):
        kinetic_x[keyi] = dt * int(keyx.split("#")[-1])
        kinetic_y[keyi] = kinetic_energy[keyx]["kinetic_energy"][0]

    assert kinetic_x.shape == potential_x.shape
    total_x = kinetic_x
    total_y = potential_y + kinetic_y

    # plot field energy and potential energy
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    field_colour = "b"
    potential_colour = "g"
    kinetic_colour = "r"
    total_colour = "k"

    ax.plot(
        x,
        y,
        color=field_colour,
        label="Field Energy",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        potential_x,
        potential_y,
        color=potential_colour,
        label="Potential Energy",
        linewidth=2,
        markersize=8,
    )

    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Field Energy: $\int_{\Omega} u^2 dx$")
    ax2.set_ylabel(
        r"Potential Energy: $\frac{1}{2}\sum_{i} \phi(\vec{r}_i)q_i$"
    )

    ax.yaxis.label.set_color(field_colour)
    ax2.yaxis.label.set_color(potential_colour)

    fig.savefig("field_energy.pdf")

    # plot potential, kinetic and total energy
    fig = plt.figure(figsize=(10, 6))
    te_ax = fig.add_subplot(111)

    pe_ax = te_ax.twinx()
    ke_ax = te_ax.twinx()
    ke_ax.spines.right.set_position(("axes", 1.15))

    pe_ax.plot(
        potential_x,
        potential_y,
        color=potential_colour,
        label="Potential Energy",
        linewidth=2,
        markersize=8,
    )
    ke_ax.plot(
        kinetic_x,
        kinetic_y,
        color=kinetic_colour,
        label="Kinetic Energy",
        linewidth=2,
        markersize=8,
    )
    te_ax.plot(
        total_x,
        total_y,
        color=total_colour,
        label="Total Energy",
        linewidth=2,
        markersize=8,
    )

    # ax.set_yscale('log')
    te_ax.set_xlabel(r"Time")
    pe_ax.set_ylabel(
        r"Potential Energy: $\mathcal{U} = \frac{1}{2}\sum_{i} \phi(\vec{r}_i)q_i$",
        color=potential_colour,
    )
    ke_ax.set_ylabel(
        r"Kinetic Energy: $\mathcal{K} = \sum_{i} \frac{1}{2}m_i|\vec{v}_i|^2 $",
        color=kinetic_colour,
    )
    te_ax.set_ylabel(
        r"Total Energy: $ \mathcal{U} + \mathcal{K} $", color=total_colour
    )

    fig.savefig("all_energy.pdf", bbox_inches="tight")
