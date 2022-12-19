import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.ticker as mtick
import h5py
import xml.etree.ElementTree as ET
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


class Session:

    def __init__(self, filename):
        """
        Helper class to read XML Nektar++ session file.
        
        :param: filename Nektar++ session file to read.
        """
        self.xml_tree = ET.parse(filename)
        self.xml_root = self.xml_tree.getroot()
        self.parameters = (
            self.xml_root.find("CONDITIONS").find("PARAMETERS").findall("P")
        )
    

    def get_parameter(self, name, t=float):
        """
        Get value from session file and cast to type.

        :param: name Name of value to extract.
        :param: t Type to cast to.
        :returns: Value cast to type.
        """
        dt = -1
        for px in self.parameters:
            text = [tx for tx in px.itertext()][0].strip()
            if text.startswith(name):
                dt = float(text.split("=")[-1])
        return t(dt)


def plot_figures(session, trajectory):
    """
    Plot the particle velocity in z as a function of time.
    """

    step_keys = sorted(trajectory.keys(), key=lambda x: int(x.split("#")[-1]))
    N = session.get_parameter("num_particles_total", int)
    dt = session.get_parameter("particle_time_step")
    dt_step = session.get_parameter("particle_num_write_particle_steps")


    assert N > 0, "bad particle count"
    step_count = len(step_keys)
    
    time_array = np.empty((step_count,))
    time_data_z = np.empty((N, step_count))
    time_data_y = np.empty((N, step_count))
    time_data_x = np.empty((N, step_count))
    
    for stepi, stepx in enumerate(step_keys):
        time_array[stepi] = stepi * dt * dt_step
        step = trajectory[stepx]
        for particle_id, velocity_z in zip(step["PARTICLE_ID_0"], step["V_2"]):
            time_data_z[particle_id, stepi] = velocity_z
        for particle_id, velocity_y in zip(step["PARTICLE_ID_0"], step["V_1"]):
            time_data_y[particle_id, stepi] = velocity_y
        for particle_id, velocity_x in zip(step["PARTICLE_ID_0"], step["V_0"]):
            time_data_x[particle_id, stepi] = velocity_x

    # plot field energy and potential energy
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)

    norm = mpl.colors.Normalize(vmin=1.0, vmax=2.0)
    
    for particle_id in range(200):
        ax.plot(
            time_array, 
            time_data_z[particle_id,:], 
            color=cm.viridis(norm(time_data_y[particle_id, 0])),
            linewidth=0.2,
        )

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    fig.colorbar(sm, label=r"$\vec{v}^{(y)}_i(t=0)$")
    # ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Velocity in $z$: $\vec{v}^{(z)}_i$")


    fig.savefig("z_velocity.pdf", bbox_inches="tight")



    # plot field energy and potential energy
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    norm = mpl.colors.Normalize(vmin=1.0, vmax=2.0)
    
    for particle_id in range(200):
        ax.plot(
            time_data_y[particle_id,:], 
            time_data_z[particle_id,:], 
            color=cm.viridis(norm(time_data_y[particle_id, 0])),
            linewidth=0.2,
        )

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    fig.colorbar(sm, label=r"$\vec{v}^{(y)}_i(t=0)$")
    # ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.set_xlabel(r"Velocity in $y$: $\vec{v}^{(y)}_i$")
    ax.set_ylabel(r"Velocity in $z$: $\vec{v}^{(z)}_i$")


    fig.savefig("vyvz_velocity.pdf", bbox_inches="tight")





if __name__ == "__main__":

    if (len(sys.argv) < 3) or ("--help" in sys.argv) or ("-h" in sys.argv):
        print(
            """
Plots energies from electrostatic PIC. Call with:

python3 plot_spiral.py input.xml Electrostatic2D3V_particle_trajectory.h5part

"""
        )
        quit()

    session = Session(sys.argv[1])
    trajectory = h5py.File(sys.argv[2], "r")
    plot_figures(session, trajectory)
