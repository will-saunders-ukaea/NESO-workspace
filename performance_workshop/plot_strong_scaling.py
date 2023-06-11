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

cores_per_node = 32
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


    files = glob.glob("*/plot.json")
    
    data = {}
    for fx in files:
        print(fx)
        meta = json.loads(open(fx).read())
        print(meta)
        num_particles = meta["size"]
        dirname = os.path.dirname(fx)
        log_files = glob.glob(os.path.join(dirname, "*.stdout.log"))

        tt = []
        nn = []
        nntt = {}
        for logx in log_files:
            t = get_time(logx);
            n = get_num_node(logx, meta)
            print(logx, n, t)
            nntt[n] = t

        for n in sorted(nntt.keys()):
            tt.append(nntt[n])
            nn.append(n)

        eff = []
        base_time = tt[0]
        for n in sorted(nntt.keys()):
            t = nntt[n]
            eff.append(((base_time / n) / t) * 100.0)

        data[num_particles] = {}
        data[num_particles]["nodes"] = nn
        data[num_particles]["times"] = tt
        data[num_particles]["eff"] = eff
    
    print(data)


    fig = plt.figure(figsize=(10, 8))
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



    ## line1_files = sys.argv[1]
    ## source_files1 = glob.glob(os.path.join(os.path.join(line1_files, "*", "slurm-*.out")))
    ## line2_files = sys.argv[2]
    ## source_files2 = glob.glob(os.path.join(os.path.join(line2_files, "*", "slurm-*.out")))
    ## 
    ## line1 = []
    ## npart_total = -1
    ## for filex in source_files1:
    ##     print(filex)
    ##     num_nodes = get_num_node(filex)
    ##     print(num_nodes)
    ##     time_taken = get_time(filex)
    ##     print(time_taken)
    ##     line1.append((num_nodes, time_taken))
    ##     npart_total = get_npart_total(filex)
    ## 
    ## line1 = sorted(line1, key=lambda x: x[0])
    ## line1 = np.array(line1)
    ## 
    ## line2 = []
    ## for filex in source_files2:
    ##     print(filex)
    ##     num_nodes = get_num_node(filex)
    ##     print(num_nodes)
    ##     time_taken = get_time(filex)
    ##     print(time_taken)
    ##     line2.append((num_nodes, time_taken))
    ## 
    ## line2 = sorted(line2, key=lambda x: x[0])
    ## line2 = np.array(line2)
    ## 
    ## 
    ## ideal = np.zeros((2,2))
    ## ideal[0,:] = line1[0,:]
    ## ideal[1,:] = line1[-1,:]
    ## ideal[1,1] = line1[0,1] / line1[-1,0]
    ## 
    ## eff1 = (ideal[0,1] / (ideal[-1,0] / ideal[0,0])) / line1[-1,1]
    ## eff2 = (line2[0,1] / (line2[-1,0] / line2[0,0])) / line2[-1,1]
    ## 
    ## # plot field energy and potential energy
    ## fig = plt.figure(figsize=(6, 5))
    ## ax = fig.add_subplot(111)
    ## 
    ## ax.plot(line2[:, 0], line2[:, 1], color="b", linestyle="--", linewidth=2, label="Large (${:.1f}$\%)".format(eff2*100))
    ## ax.plot(line1[:, 0], line1[:, 1], color="r", label="Small (${:.1f}$\%)".format(eff1*100))
    ## ax.plot(ideal[:, 0], ideal[:, 1], color="k", linestyle=":", label="Ideal Scaling")
    ## 
    ## ax.set_xscale("log")
    ## ax.set_yscale("log")
    ## ax.legend()
    ## 
    ## ax.set_ylabel(r"Time per step (s)")
    ## ax.set_xlabel(r"Number of Nodes")
    ## 
    ## ax.set_xticks(line1[:,0])
    ## ax.set_xticklabels([format_node_count(cx) for cx in line1[:,0]])
    ## #ax.minorticks_off()
    ## 
    ## ax2 = ax.twiny()
    ## ax2.set_xscale('log')
    ## ax2.set_xlim(ax.get_xlim())
    ## ax2.set_xticks(line1[:, 0])
    ## ax2.set_xticklabels([prettyInt(npart_total / (cx * cores_per_node)) for cx in line1[:,0]])
    ## ax2.set_xlabel(rf"Particles per core")
    ## 
    ## ax.tick_params(axis='x', which='minor', top=False, bottom=False, labelbottom=False, labeltop=False)
    ## ax2.tick_params(axis='x', which='minor', top=False, bottom=False, labelbottom=False, labeltop=False)
    ## 
    ## fig.savefig("two_stream_strong_scaling.pdf", bbox_inches="tight")

