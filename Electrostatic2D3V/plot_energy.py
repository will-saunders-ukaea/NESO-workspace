import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import h5py

plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('text', usetex=True)
plt.rc('savefig', dpi=500)
plt.rc('xtick', labelsize='small')
plt.rc('legend', fontsize='medium')
plt.rc('legend', edgecolor='lightgray')
plt.rc('ytick', labelsize='small')
plt.rcParams.update({'font.size': 20})

if 'poster' in sys.argv:
    print("Using poster font")
    import matplotlib.font_manager as mfm
    print(mfm.findfont('Lato'))
    plt.rc('font', family='sans-serif', serif='Lato')
    params = {'text.latex.preamble': [r'''
    \usepackage{amsmath}
    \usepackage{cmbright}
    \usepackage[default]{lato}
    ''']}
else:
    plt.rc('font', family='serif', serif='Times New Roman')
    params = {'text.latex.preamble': r"\usepackage{amsmath}"}

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


if __name__ == '__main__':

    
    energy_h5 = h5py.File(sys.argv[1], "r")
    
    keys = sorted(energy_h5.keys(), key=lambda x: int(x.split("#")[-1]))


    N = len(keys)
    x = np.zeros((N,))
    y = np.zeros((N,))
    for keyi, keyx in enumerate(keys):
        # TODO write dt, step number
        x[keyi] = 0.001 * keyi * 20
        y[keyi] = energy_h5[keyx]["field_energy"][0]

 
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(x, y, color='b', label='Energy', linewidth=2,markersize=8)

    #ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Field Energy: $\int_{\Omega} u^2 dx$')

    #ax2 = ax.twiny()
    #ax2.set_axes_locator(divider.new_locator(nx=1, ny=1))
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    #ax2.set_xscale('log')
    #ax2.set_xlim(ax.get_xlim())
    #ax2.set_xlabel('Charges per node',labelpad=20)
    #ax2.set_xticks(xtick_locs)
    #ax2.set_xticklabels(top_labels, rotation=30)
    #ax2.tick_params(axis='x',which='minor', top=False)

    #ax.tick_params(axis='x', which='minor', bottom=False)
    #ax.set_xticks(xtick_locs)
    #ax.set_xticklabels(xtick_values, minor=False)


    fig.savefig('energy.pdf')#, bbox_inches='tight')









