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
import json

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

def format_exp(v):
    return '{: 2.1e}'.format(v)


def prettyInt(x, latex=True):
    if (latex):
        s = ('%6.1e' % x)
        abscissa, exponent = s.split('e')
        return r'$'+abscissa+r'\cdot 10^{'+str(int(exponent))+r'}$'
    else:
        s = ('%4.1f' % (1.E-6*x))+'mio'
        return s

cores_per_node = 128
def format_node_count(count):
    if count.is_integer():
        return "$" + str(int(count)) + "$"
    else:
        return str(round(count * cores_per_node)) + "/" + str(cores_per_node)

def get_num_node(path, meta):
    num_nodes = meta[os.path.basename(path)]
    return num_nodes

def get_time(path):
    with open(path) as fh:
        for linex in fh:
            if linex.startswith("BENCHMARK Time taken per step"):
                return float(linex.split()[-1])

def get_npart_total(path):
    with open(path) as fh:
        for linex in fh:
            if linex.startswith("Particle count"):
                return int(linex.split()[-1])

if __name__ == "__main__":
    
    data = {}

    nn = [
        1,
        2,
        4,
        8,
        16,
        32,
    ]

    num_particles = 1E8
    base_time = 0.417254
    tt = [
        0.417254,
        0.211533,
        0.109097,
        0.0565779,
        0.0313048,
        0.0191582,
    ]
    eff = [((base_time / nx) / tx) * 100.0 for nx, tx in zip(nn, tt)]
    data[num_particles] = {}
    data[num_particles]["nodes"] = nn
    data[num_particles]["times"] = tt
    data[num_particles]["eff"] = eff

    num_particles = 4E8
    base_time = 1.6745
    tt = [
        1.6745,
        0.836638,
        0.428597,
        0.21695,
        0.112194,
        0.0601244,
    ]
    eff = [((base_time / nx) / tx) * 100.0 for nx, tx in zip(nn, tt)]
    data[num_particles] = {}
    data[num_particles]["nodes"] = nn
    data[num_particles]["times"] = tt
    data[num_particles]["eff"] = eff
    
    print(data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    for px in sorted(data.keys()):
        ax.plot(
            data[px]["nodes"], 
            data[px]["times"], 
            label=prettyInt(px) + " particles",
            marker='*'
        )
        for ix, nx in enumerate(data[px]["nodes"]):
            tx = data[px]["times"][ix]
            ex = data[px]["eff"][ix]
            ax.annotate("{:2.1f}%".format(ex), (nx, tx))

    largest = max(data.keys())
    max_nodes = max(data[largest]["nodes"])
    
    ideal = np.zeros((2,2))
    ideal[0,0] = 1
    ideal[1,0] = max_nodes

    max_time = data[largest]["times"][0]
    ideal[0,1] = max_time
    ideal[1,1] = max_time / max_nodes
    

    # ax.plot(ideal[:, 0], ideal[:, 1], color="k", linestyle=":", label="Ideal Scaling")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    
    ax.set_ylabel(r"Time per step (s)")
    ax.set_xlabel(r"Number of Nodes")
    
    ax.set_xticks(data[largest]["nodes"][:])
    ax.set_xticklabels([str(cx) for cx in data[largest]["nodes"][:]])
    #ax.minorticks_off()
    
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(data[largest]["nodes"][:])
    ax2.set_xticklabels([prettyInt(largest / (cx * cores_per_node)) for cx in data[largest]["nodes"][:]])
    ax2.set_xlabel(rf"Particles per core")
    
    ax.tick_params(axis='x', which='minor', top=False, bottom=False, labelbottom=False, labeltop=False)
    #ax2.tick_params(axis='x', which='minor', top=False, bottom=False, labelbottom=False, labeltop=False)
    
    fig.savefig("strong_scaling.png", bbox_inches="tight")
    fig.savefig("strong_scaling.pdf", bbox_inches="tight")


