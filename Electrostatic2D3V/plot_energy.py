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
        self.xml_tree = ET.parse(filename)
        self.xml_root = self.xml_tree.getroot()
        self.parameters = self.xml_root.find("CONDITIONS").find("PARAMETERS").findall("P")

    def get_parameter(self, name, t=float):
        dt = -1
        for px in self.parameters:
            text = [tx for tx in px.itertext()][0].strip()
            if text.startswith(name):
                dt = float(text.split("=")[-1])
        return t(dt)



if __name__ == "__main__":

    if (len(sys.argv) < 5) or ("--help" in sys.argv) or ("-h" in sys.argv):
        print(
            """
Plots energies from electrostatic PIC. Call with:

python3 plot_energy.py input.xml field_energy.h5 potential_energy.h5 kinetic_energy.h5

"""
        )
        quit()
    
    session = Session(sys.argv[1])
    dt = session.get_parameter("particle_time_step")
    assert dt > 0.0


    num_particles_total = session.get_parameter("num_particles_total")
    q = session.get_parameter("particle_charge_density") / num_particles_total
    print("q", q)
    m = 1.0;
    print("m", m)
    n = session.get_parameter("particle_number_density")
    print("n", n, (16.0 * math.pi * math.pi / (3)))
    epsilon_0 = 1.0
    print("epsilon_0", epsilon_0)
    v_b = session.get_parameter("particle_initial_velocity")
    print("v_b", v_b)
    PI_s = math.sqrt((q*q*n) / (m * epsilon_0))
    print("PI_s", PI_s)

    
    # sqrt(3)/2 = v_b * k_parallel / PI_s
    k_parallel = math.sqrt(3) * 0.5 * PI_s / v_b
    print("k_parallel", k_parallel, "v_b k_par,max", v_b * k_parallel, "Pi_s sqrt(3)/2", PI_s * math.sqrt(3)/2)
    u = v_b * k_parallel / PI_s
    x_minus = (u*u + 1.0 - (4.0 * u * u + 1.0)**0.5)**0.5
    x_plus = (u*u + 1.0 + (4.0 * u * u + 1.0)**0.5)**0.5

    x_mm = -x_minus
    x_pm = x_minus
    x_pp = x_plus
    x_mp = -x_plus

    print("x--", x_mm)
    print("x+-", x_pm)
    print("x++", x_pp)
    print("x-+", x_mp)
    print("sqrt(15)/2", 15**0.5 / 2)

    omega_mm = x_mm * PI_s
    omega_pm = x_pm * PI_s
    omega_pp = x_pp * PI_s
    omega_mp = x_mp * PI_s
 
    print("omega_mm", omega_mm)
    print("omega_pm", omega_pm)
    print("omega_pp", omega_pp)
    print("omega_mp", omega_mp)

    print("PI_s                     :", PI_s)
    print("PI_s: v_b 4pi/sqrt(3)    :", v_b * 4.0 * math.pi / math.sqrt(3))
    print("n                        :", n)
    print("v_b^2 (m/q^2) 16 pi^2/3  :", v_b*v_b * (m/(q*q)) * 16 * math.pi**2  / 3.0)


    gamma_max = PI_s / 2
    PI_T = PI_s * math.sqrt(2)
    print("PI_s/2                   :", PI_s/2)
    print("v_b M 2pi/sqrt(3)        :", v_b * 2.0 * math.pi / math.sqrt(3))
    print("PI_T/(2sqrt(2))          :", PI_T / (2.0 * math.sqrt(2)))
    print("(1/2) * sqrt(32pi**2/6)  :", 0.5 * math.sqrt(((32 * (math.pi ** 2))) / 6))

    #gamma_max = 0.5 * math.sqrt(((32 * (math.pi ** 2))) / 6)
    print("gamma_max", gamma_max, 0.5 * math.sqrt(((32 * (math.pi ** 2))) / 6))
    
    
    w = v_b**2 * 16 * (math.pi**2 / 3) * (2 / num_particles_total)
    print("w", w, 32 * math.pi**2 / (3 * num_particles_total) )
    nT = 32 * math.pi **2 / 3
    print("nT", nT)

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

    total_initial = total_y[0]
    total_end = total_y[-1]

    print("Energy diff:", abs(total_initial - total_end) / abs(total_initial))
    
    potential_energy_max_index = np.argmax(potential_y)
    potential_energy_min_index = np.argmin(potential_y)

    dx = potential_x[potential_energy_max_index] - potential_x[potential_energy_min_index]
    dy = np.log(potential_y[potential_energy_max_index]) - np.log(potential_y[potential_energy_min_index])

    print("Gradient:", dy / dx)
    print("WARNING +-1 in X AND Y")
    

    tx0 = potential_x[potential_energy_min_index]
    tx1 = potential_x[potential_energy_max_index]


    print("WARNING gamma *= sqrt(2)")
    gamma_max *= math.sqrt(2)
    iy0 = math.exp(gamma_max * tx0)
    iy1 = math.exp(gamma_max * tx1)

    ishift = 1.0


    # plot field energy and potential energy
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    field_colour = "b"
    potential_colour = "m"
    kinetic_colour = "r"
    total_colour = "k"

    ax.semilogy(
        x,
        y,
        color=field_colour,
        label="Field Energy",
        linewidth=2,
        markersize=8,
        base=np.e
    )
    ax2.semilogy(
        potential_x,
        potential_y,
        color=potential_colour,
        label="Potential Energy",
        linewidth=2,
        markersize=8,
        base=np.e
    )
    ax2.semilogy(
        [potential_x[potential_energy_min_index], potential_x[potential_energy_max_index]],
        [potential_y[potential_energy_min_index], potential_y[potential_energy_max_index]],
        color="r",
        label="Fit",
        linewidth=2,
        markersize=8,
        linestyle="--",
        base=np.e
    )

    ax2.semilogy(
        [tx0, tx1],
        [ishift*iy0, ishift*iy1],
        color="k",
        label="Fit",
        linewidth=2,
        markersize=8,
        linestyle="--",
        base=np.e
    )


    #ax.set_yscale("log")
    #ax2.set_yscale("log")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Field Energy: $\int_{\Omega} \phi^2 dx$")
    ax2.set_ylabel(
        r"Potential Energy: $\frac{1}{2}\sum_{i} \phi(\vec{r}_i)q_i$"
    )

    def tick_formater(y, pos):
        return r'$e^{{{:.0f}}}$'.format(np.log(y))

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(tick_formater))
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(tick_formater))

    ax.yaxis.label.set_color(field_colour)
    ax2.yaxis.label.set_color(potential_colour)

    fig.savefig("field_energy.pdf")

    # plot potential, kinetic and total energy
    fig = plt.figure(figsize=(10, 7))
    te_ax = fig.add_subplot(111)

    pe_ax = te_ax.twinx()
    ke_ax = te_ax.twinx()
    ke_ax.spines.right.set_position(("axes", 1.15))

    pe_ax.plot(
        potential_x,
        (potential_y - potential_y[0]) / abs(total_initial),
        color=potential_colour,
        label="Potential Energy",
        linewidth=2,
        markersize=8,
    )
    ke_ax.plot(
        kinetic_x,
        (kinetic_y - kinetic_y[0]) / abs(total_initial),
        color=kinetic_colour,
        label="Kinetic Energy",
        linewidth=2,
        markersize=8,
    )
    te_ax.plot(
        total_x,
        np.abs(total_y - total_initial) / abs(total_initial),
        color=total_colour,
        label="Total Energy",
        linewidth=2,
        markersize=8,
    )

    # ax.set_yscale('log')
    te_ax.set_xlabel(r"Time")
    pe_ax.set_ylabel(
        r"$(E_{\mathcal{U}} - E(0)) / E(0)$, $E_{\mathcal{U}} = \frac{1}{2}\sum_{i} \phi(\vec{r}_i)q_i$",
        color=potential_colour,
    )
    ke_ax.set_ylabel(
        r"$(E_{\mathcal{K}} - E(0)) / E(0)$, $E_{\mathcal{K}} = \sum_{i} \frac{1}{2}m_i|\vec{v}_i|^2 $",
        color=kinetic_colour,
    )
    te_ax.set_ylabel(
        r"Total Energy ($E$) Rel. Error: $|E - E(0)|/|E(0)|$", color=total_colour
    )

    fig.savefig("all_energy.pdf", bbox_inches="tight")
