import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


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
    params = {'text.latex.preamble': [r'''
    \usepackage{amsmath}
    ''']}


plt.rcParams.update(params)


def format_exp(v):
    return '{: 2.1e}'.format(v)

def get_cores_per_node(directory):
    with open(os.path.join(directory, 'cores_per_node')) as fh:
        c = int(fh.read())
    return c

def get_platform_title(directory):
    with open(os.path.join(directory, 'platform_title')) as fh:
        c = fh.read().splitlines()[0]
    return c

def get_linespec(directory):
    with open(os.path.join(directory, 'linespec')) as fh:
        c = fh.read().splitlines()[0]
    return c



class FromDict:
    def __init__(self, d):
        for ix in d.items():
            setattr(self, ix[0], ix[1])


class Run(FromDict):
    def __repr__(self):
        s = ''
        s += '-'*60 + '\n'
        s += "Platform: {}\n".format(self.platform_title)
        s += "Cores per node: {}\n".format(self.cores_per_node)
        s += '.'*60 + '\n'
        s += 'nnode    Time (s)\n'
        s += '.'*60 + '\n'
        for rx in self.results:
            s += "{:4d}    {}\n".format(rx.nnode, rx.time)
        s += '-'*60 + '\n'
        return s
    
    def list(self, p):
        x = []
        for rx in self.results:
            x.append(getattr(rx, p))
        return x

    @property
    def list_time(self):
        x = []
        for rx in self.results:
            x.append(rx.time)
        return x

    def __lt__(self, other):
        return self.platform_title < other.platform_title


class Result(FromDict):
    def __lt__(self, other):
        return self.ncore < other.ncore
    def __eq__(self, other):
        return self.ncore == other.ncore



def get_result(filename, cores_per_node):
    
    params = os.path.basename(filename).split('_')

    nproc   = int(params[1])
    nthread = int(params[2])
    ncharge = int(params[3])
    nstep   = int(params[4].split('.')[0])

    t = np.inf
    with open(filename) as fh:
        for lx in fh:
            tp = float(lx)
            t = min(t, tp)

    t /= nstep
    
    ncore = nproc * nthread
    nnode = int(ncore // cores_per_node)

    return Result({
        'ncore': ncore,
        'nnode': nnode,
        'nproc': nproc,
        'nthread': nthread,
        'nstep': nstep,
        'ncharge': ncharge,
        'time': t
    })



def compute_efficiency(raw):

    basetime = raw[0].time
    basennode = raw[0].nnode
    raw[0].node_efficiency = 100.0

    for rx in raw[1:]:
        rx.node_efficiency = 100.0 * ((basetime / (rx.nnode / basennode)) / rx.time)
 

def skip(directory):
    return os.path.exists(os.path.join(directory, 'skip'))


def get_ideal(run, x_prop, y_prop):

    xlist = run.list(x_prop)
    ylist = run.list(y_prop)

    x = (xlist[0], xlist[-1])
    y = (ylist[0], ylist[0] / (x[1]/x[0]))

    return x, y



def prettyInt(x, latex=True):
    if (latex):
        s = ('%6.1e' % x)
        abscissa, exponent = s.split('e')
        return r'$'+abscissa+r'\cdot 10^{'+str(int(exponent))+r'}$'
    else:
        s = ('%4.1f' % (1.E-6*x))+'mio'
        return s




if __name__ == '__main__':
 
    res_dirs_raw = glob.glob('./*')
    res_dirs = [dx for dx in res_dirs_raw if os.path.isdir(dx)]
    res_dirs.sort()
    
    raw = []



    # collect raw results
    for dx in res_dirs:
        if skip(dx):
            continue

        key = os.path.basename(dx)

        rdx = []

        cores_per_node = get_cores_per_node(dx)
        platform_title = get_platform_title(dx)
        linespec = get_linespec(dx)

        files = glob.glob(os.path.join(dx, '*.result'))
        for fx in files:
            r = get_result(fx, cores_per_node)
            rdx.append(r)


        # select the best times
        r2 = {}
        for rx in rdx:
            if rx.nnode not in r2.keys():
                r2[rx.nnode] = rx
            else:
                if (r2[rx.nnode].time > rx.time):
                    r2[rx.nnode] = rx
        
        rdx = list(r2.values())
        rdx.sort()

        raw.append(
            Run({
                'platform_title': platform_title,
                'cores_per_node': cores_per_node,
                'linespec': linespec,
                'results': rdx
            })
        )

    raw.sort()

    xticks = set()
    # compute metrics
    for rx in raw:
        compute_efficiency(rx.results)
        xticks = xticks.union(rx.list('nnode'))


    xtick_locs = list(xticks)
    xtick_values = [r'${}$'.format(xx) for xx in xtick_locs]

    for rx in raw:
        print(rx)
    
    
    with open('plot_title') as fh:
        plot_title = fh.read()


    with open('ncharges') as fh:
        ncharges = int(fh.read())
    
    
    top_labels = [prettyInt(int(ncharges / xx)) for xx in xtick_locs]

    if "paper" in sys.argv:
        fig = plt.figure(figsize=(8.3, 6))
    else:
        fig = plt.figure(figsize=(6, 6))


    ax = fig.add_subplot(111)

    h = [Size.Fixed(1.0), Size.Fixed(4.5)]
    v = [Size.Fixed(0.7), Size.Fixed(4.)]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
    # the width and height of the rectangle is ignored.

    #ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    
    idx, idy = get_ideal(raw[-3], 'nnode', 'time')
    ax.plot(idx, idy, color='r', label='Ideal Scaling', linestyle=':',linewidth=2,markersize=8)
    
    for rx in raw:
        ax.plot(rx.list('nnode'), rx.list('time'), rx.linespec, label=rx.platform_title,linewidth=2,markersize=8)





    ax.set_xscale('log')
    ax.set_yscale('log')
    
    #ylim = ax.get_ylim()
    #ax.set_ylim((ylim[0], 40))

    #ylocs = ax.get_yticks()
    #ylabels = [format_exp(yy) for yy in ylocs]
    #ax.set_yticklabels(ylabels)
    #ax.set_title(titles[dx])
    


    
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())



    if "paper" in sys.argv:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=False)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01), fancybox=False)



    ax.set_xlabel(r'Number of Nodes')
    ax.set_ylabel(r'Time per KMC step [$s$]')
    ax.set_xlim(1./1.2,128*1.2)
    ax.set_ylim(0.08,75)


    ax2 = ax.twiny()
    ax2.set_axes_locator(divider.new_locator(nx=1, ny=1))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Charges per node',labelpad=20)
    ax2.set_xticks(xtick_locs)
    ax2.set_xticklabels(top_labels, rotation=30)
    ax2.tick_params(axis='x',which='minor', top=False)

    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_values, minor=False)


    if 'poster' in sys.argv:
        print("saving for poster")
        fig.savefig('kmc_strong_scaling.pdf', transparent=True)
    else:
        fig.savefig('kmc_strong_scaling.pdf')#, bbox_inches='tight')





    # latex table outputting

    table = np.zeros((len(xticks), len(raw) + 1, 2), 'float64')
    table[:, :, 0] = np.nan
    c0 = list(xticks)
    c0.sort()
    table[:, 0, 0] = c0

    for rxi, rx in enumerate(raw):
        nnode = rx.list('nnode')
        times = rx.list('time')
        eff = rx.list('node_efficiency')

        for nxi, nx in enumerate(nnode):
            loc = np.where(nx == table[:, 0])[0]
            table[loc, rxi+1, 0] = times[nxi]
            table[loc, rxi+1, 1] = eff[nxi]
    
    
    def to_format(f, e):
        print(f, e)
        if np.isnan(f):
            return ' & '
        else:
            return r'{:8.2f} & ({:.1f}\%)'.format(f, e)

    
    middle = ''
    for rowxi in range(table.shape[0]):
        rowx = table[rowxi,:, 0]
        rowxf = table[rowxi,:, 1]

        middle += '{:4d} &'.format(int(rowx[0]))
        middle += ' & '.join([to_format(tx, ex) for tx, ex in zip(rowx[1:], rowxf[1:])]) + '\\\\\n'

    
    format_spec = 'r' + '|rr'*len(raw)
    header =   '                 & \\multicolumn{{{NUM_RAW}}}{{c}}{{Time per KMC step [s]}}\\\\\n'.format(NUM_RAW=len(raw) * 2)
    header += '\\cline{{2-{}}}\n'.format(len(raw)*2 + 1)
    header += r' $P$          & ' + ' & '.join(['\\multicolumn{{2}}{{c}}{{{}}}'.format(rx.platform_title) for rx in raw])




    frame = r"""
\begin{{table*}}
\begin{{center}}
  \begin{{tabular}}{{{FORMAT_SPEC}}}
    \hline
    {HEADER}\\
    \hline
    \hline
{MIDDLE}
    \hline
  \end{{tabular}}
  \caption{{TODO}}
\label{{tab:kmc_strong_scaling}}
\end{{center}}
\end{{table*}}
""".format(
        FORMAT_SPEC=format_spec,
        HEADER=header,
        MIDDLE=middle
    )

print(frame)

with open('latex_table', 'w+') as fh:
    fh.write(frame)















