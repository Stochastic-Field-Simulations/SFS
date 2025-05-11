import h5py
import numpy as np
from pathlib import Path

import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation as FA
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=False)
plt.rc("grid", linestyle="--", alpha=1)

uni_to_asc = {
    "varphi" : "φ", "varphi_1" : "φ_1", "varphi_2" : "φ_2", "varphi_3" : "φ_3", "varphi_4" : "φ_4"
    }


def creat_map(name, color, alpha):
    ncolors = 256
    color_array = np.zeros((ncolors, 4))
    color_array[:,-1] = np.linspace(.0,alpha,ncolors)
    color_array[:,0:3] = np.array(color)[None, :]
    map_object = LinearSegmentedColormap.from_list(name=name,colors=color_array)
    plt.colormaps.register(cmap=map_object)

creat_map("blue", (.2, .2, .8), .8)
creat_map("red", (.8, .2, .2), .8)


# Load data

def count_files(folder):
    path = Path(folder)
    return sum(1 for _ in path.glob('*'))


def get_para(folder):
    para = h5py.File(folder+"parameters") 
    con = para["con"]
    
    try:     
        tools = para["tools"]
        sys = tools["sys"]
        d, N, L, T, dt = sys[["d", "N", "L", "T", "Δt"]]
        
        # x is only saved as a reference in tools
        x = para[tools["x"]]
        # This reference again points to an array of references,
        # which each points to the coordinate array for its dimension
        xi = [np.array(para[x[i]]).reshape(N) for i in range(d)]
        # we then blow this up for plotting purposes
        xi = np.meshgrid(*xi)
    except: # dirty hack for when tools is not saved
        xi, d, N, L, T, dt = 0, 0, 0, 0, 0, 0

    return xi, d, N, L, T, dt, con


def get_field(folder, fn):
    if fn in uni_to_asc.keys(): fn = uni_to_asc[fn]
    data = h5py.File(folder+fn)
    return np.array([data[k] for k in data.keys()])


def get_time(folder):
    times = h5py.File(folder+"TIME")
    return np.array([times[k][()] for k in times.keys()])


# Plotting

def plot_field(ax, d, X, field, norm=[-1,1], colorbar=False, label='', cmap=cm.viridis, **kw):
    if d==1:
        ax.set_ylim(norm)
        return ax.plot(X[0], field, **kw)[0]
    elif d==2: 
        pl = ax.pcolormesh(X[0], X[1], field, vmin=norm[0], vmax=norm[1], label=label, cmap=cmap, **kw)
        if colorbar: ax.figure.colorbar(pl, ax=ax, label=label,)
        return pl


def update_plot(l, d, field):
    if d==1: return l.set_ydata(field)
    elif d==2: return l.set_array(field.ravel())


def get_norm(field):
    mm = [np.min(field), np.max(field)]
    d = (mm[1] - mm[0]) / 2
    if d is np.nan: d = 1.
    mm[0] -= d * .1
    mm[1] += d * .1
    return mm


def plot_anim(anim, path, SAVE, **kwargs):
    if SAVE: print("Saving..."); anim.save(path)
    else: plt.show()


def plot(fig, path, SAVE, **kwargs):
    if SAVE: fig.savefig(path, **kwargs)
    else: plt.show()


def get_para_folder(folder, para_folder):
    if para_folder is None: return get_para(folder)
    else: 
        xi, d, N, L, T, dt = get_para(para_folder)[:-1]
        con = get_para(folder)[-1]
        return xi, d, N, L, T, dt, con 


def anim_fields(
    folder, 
    SAVE=False, ax_lst=None, interval=1, rows=1, para_folder=None,
    name="vid", skip=1, fns=["varphi", ], size=6, lim=None, **kw
    ):
    fields = [get_field(folder, fn) for fn in fns] 
    para = get_para_folder(folder, para_folder)
    return anim_fields_array(fields, para, 
        SAVE=SAVE, ax_lst=ax_lst, interval=interval, 
        name=name, skip=skip, fns=fns, size=size, lim=lim, **kw
    )


def anim_many_fields(
    folders,
    SAVE=False, seed=0, ax_lst=None, interval=50, cmap=cm.viridis, rows=1,
    name="vid", skip=1, fns=["varphi", ], size=6, lim=None, M=None, **kw
    ):

    fields_arr = [[get_field(folder, fn) for fn in fns] for folder in folders] 
    para = get_para(folders[0])
    X, d, N, L, T, dt, con = para

    if M is None: M = len(fields_arr[0][0])
    Nf = len(fns)

    ar = 1.2
    nn = len(fields_arr)
    fig, ax = plt.subplots(rows, nn//rows, figsize=(size*nn/rows*ar, size*rows))
    ax = ax.flatten()
    [a.yaxis.set_ticklabels([]) for a in ax]
    [a.xaxis.set_ticklabels([]) for a in ax]

    l = []
    for k in range(nn):
        fields = fields_arr[k]
        for i in range(Nf):
            if (lim is None): mm = get_norm(fields[i])
            else: mm = lim
            if isinstance(cmap, list): color = cmap[i]
            else: color = cmap

            li = plot_field(ax[k], d, X, fields[i][0], norm=mm, cmap=color, lw=1, **kw)
            l.append(li)

    def anim(n):
        n = skip * n
        ll = []
        for k in range(nn):
            fields = fields_arr[k]
            for i in range(Nf):
                y = fields[i][n]
                ll.append(update_plot(l[k*Nf+i], d, y))
        return ll

    a = FA(fig, anim, interval=interval, frames=M//skip)

    return a, fig


def anim_fields_array(
    fields, para, 
    SAVE=False, ax_lst=None, interval=1, cmap=cm.viridis,
    name="vid", skip=1, fns=["varphi", ], size=6, lim=None, M=None, sqr=False, **kw
    ):
    X, d, N, L, T, dt, con = para
    if M is None: M = len(fields[0])
    Nf = len(fns)
    fields = np.array(fields)

    if sqr: fields = np.concatenate([fields, [np.sqrt(np.sum(fields[0:2]**2, axis=0)), ]]); Nf+=1

    if ax_lst is None: ax_lst = [i for i in range(Nf)]
    Nax = np.max(ax_lst)+1

    ar = 1.2
    fig, ax = plt.subplots(1, Nax, figsize=(size*Nax*ar, size))
    ax = np.atleast_1d(ax)

    # fields = np.array([[np.roll(f0, 64) for f0 in f] for f in fields])

    l = []
    for i in range(Nf):
        if (lim is None): mm = get_norm(fields[i])
        else: mm = lim
        if isinstance(cmap, list): color = cmap[i]
        else: color = cmap
        li = plot_field(ax[ax_lst[i]], d, X, fields[i][0], norm=mm, cmap=color, lw=1, **kw)
        l.append(li)

    def anim(n):
        n = skip * n
        ll = []
        for i in range(Nf):
            y = fields[i][n]
            ll.append(update_plot(l[i], d, y))
        
        return ll 

    a = FA(fig, anim, interval=interval, frames=M//skip)

    name = "plot/fig/" + name + ".mp4"
    
    return a, fig


def anim_fields_defect(
    folder, SAVE=False, seed=0, ax_lst=None, interval=1, 
    name="vid", skip=1, fns=["varphi", ], **kw
    ):
    
    X, d, N, L, T, dt, con = get_para(folder)

    fields = [get_field(folder, fn) for fn in fns]
    M = len(fields[0])
    Nf = len(fns)


    if ax_lst is None: ax_lst = [i for i in range(Nf)]
    Nax = np.max(ax_lst)+1

    size = 6
    ar = 1.2
    fig, ax = plt.subplots(1, Nax, figsize=(size*Nax*ar, size))
    ax = np.atleast_1d(ax)

    fig.suptitle("$r = {r}$".format(r = con["r"]))

    l = []

    fields = np.array([[np.roll(f0, 64) for f0 in f] for f in fields])

    for i in range(Nf):
        mm = get_norm(fields[i])
        li = plot_field(ax[ax_lst[i]], d, X, fields[i][0], norm=mm, lw=1)
        l.append(li)

    sp = .005
    t0 = -4.5
    m = 4

    N = len(X[0])
    x0 = m * (2 * np.pi * X[0] / L)
    x = np.tanh(x0) * x0
    x1 = m * (2 * np.pi * X[0] / L - np.pi)
    x[N//4:] = (np.tanh(-x1)*x1)[N//4:]

    # l2 = ax[0].plot(X[0], np.sin( x + t0),'k--',lw=1)[0]
    # l.append(l2)
    # l2 = ax[0].plot(X[0], np.cos( x + t0),'r--',lw=1)[0]
    # l.append(l2)

    smooth = 0
    def anim(n):
        n = skip * n
        ll = []
        for i in range(Nf):

            if smooth:
                y = np.zeros_like(fields[i][n]) 
                s = max(0, n-smooth)
                for ii in range(s, n):
                    y += fields[i][ii] / (n-s)
            else:
                y = fields[i][n]
            ll.append(update_plot(l[i], d, y))

        # l2 = l[-2].set_ydata(np.sin( x + sp * n + t0))
        # ll.append(l2)
        # l2 = l[-1].set_ydata(np.cos( x + sp * n + t0))
        # ll.append(l2)
        
        return ll

    a = FA(fig, anim, interval=interval, frames=M//skip)

    name = "plot/fig/" + name + ".mp4"
    plot_anim(a, name,  SAVE, **kw)


def anim_1D(
    folder, SAVE=False, seed=0, ax_lst=None, interval=1, 
    name="vid", skip=1, fns=["varphi", ], M=None, **kw
    ):
    
    X, d, N, L, T, dt, con = get_para(folder)

    fields = [get_field(folder, fn) for fn in fns]
    if M is None: M = len(fields[0])
    Nf = len(fns)

    if ax_lst is None: ax_lst = [i for i in range(Nf)]
    Nax = np.max(ax_lst)+1 

    size = 6
    ar = 1.2
    fig, ax = plt.subplots(1, Nax, figsize=(size*Nax*ar, size))
    ax = np.atleast_1d(ax)

    fig.suptitle("$r = {r}$".format(r = con["r"]))

    ks = [1, 32, 63]
    l = []

    for j, k in enumerate(ks):
        for i in range(Nf):
            # mm = get_norm(fields[i])
            mm = [-1.2, 1.2]
            fs = fields[i][0][k]
            li = ax[ax_lst[i]].plot(X[0][0], fields[i][0][k], color=cm.viridis(j/len(ks)), lw=.2)
            ax[ax_lst[i]].set_ylim(mm)
            l.append(li)

        li = ax[0].plot(X[0][0], np.sqrt( fields[0][0][k]**2 + fields[1][0][k]**2), color=cm.viridis(j/len(ks)), lw=.5)
        l.append(li)

    smooth = 10
    def anim(n):
        n = skip * n
        for j, k in enumerate(ks):

            s = max(1, n-smooth)
            
            for i in range(Nf):
                y = np.zeros_like(fields[i][n][k]) 
                for ii in range(s, n):
                    y += fields[i][ii][k] / (n-s)
                l[j * (Nf) + i][0].set_ydata(y)



    a = FA(fig, anim, interval=interval, frames=M//skip)

    name = "plot/fig/" + name + ".mp4"
    return plot_anim(a, name, SAVE, **kw)


def anim_bub(
    folder, SAVE=False, seed=0, ax_lst=[0, 0], interval=1, 
    name="vid", skip=1, fns=["varphi", ], M=None, **kw
    ):
    
    X, d, N, L, T, dt, con = get_para(folder)

    fields = [get_field(folder, fn) for fn in fns]
    if M is None: M = len(fields[0])
    Nf = len(fns)

    if ax_lst is None: ax_lst = [i for i in range(Nf)]
    Nax = np.max(ax_lst)+1
    size = 6
    ar = 1.2
    fig, ax = plt.subplots(1, Nax, figsize=(size*Nax*ar + 1, size))
    ax = np.atleast_1d(ax)

    fig.suptitle(
        "$\\mathcal{C}_{1 \\rightarrow 2}"+"={},".format(
            -con["α"][0]*con["r"][1] / 2 ) \
            + "\\, \\mathcal{C}_{2 \\rightarrow 1}"\
            +"={}".format(-con["α"][1]*con["r"][0] / 2 ) + "$"
            )

    l = []
    cmaps = [cm.Reds, cm.Blues, cm.viridis_r]
    labels = ["$S_1$", "S_2", "$c$"]
    zorder = [1, 2, 0]
    alphas = [.8, .8, .6]
    shift = 0
    for i in range(Nf):
        mm = get_norm(fields[i])
        y = np.roll(fields[i][0], shift, axis=1)
        if fns[i][:6]=='varphi': y = np.ma.masked_array(y, y<0.)
        li = plot_field(
            ax[ax_lst[i]], d, X, y, 
            norm=mm, lw=1, alpha=alphas[i], cmap=cmaps[i], 
            colorbar=(i==2), zorder=zorder[i], label=labels[i]
            )
        # if i==3: ax.figure.colorbar(li, ax=ax)
        l.append(li)

    smooth = 1
    def anim(n):
        n = skip * n
        ll = []
        for i in range(Nf):
            y = np.roll(fields[i][n], shift, axis=1)
            if fns[i][:6]=='varphi': y = np.ma.masked_array(y, y<0.)
            ll.append(update_plot(l[i], d, y))
        return ll

    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
    ax[0].set_xticks(np.linspace(0, L, 5))
    ax[0].set_yticks(np.linspace(0, L, 5))

    a = FA(fig, anim, interval=interval, frames=M//skip)

    name = "plot/fig/" + name + ".mp4"
    if SAVE: plot_anim(a, name,  SAVE, **kw)
    return a, fig


def vid_notebook(folder, i, skip=10, fns=["varphi", ], SAVE=False, size=4, name='vid', **kw):
    sub_folder =  folder+"{n}/".format(n=i+1)
    a, figv = anim_fields(sub_folder, SAVE=SAVE, interval=100, skip=skip, size=size, fns=fns, **kw)
    name = "fig/" + name + ".mp4"
    if SAVE: plot_anim(a, name,  SAVE, **kw)
    figv.tight_layout()
    display(HTML(a.to_jshtml()))
    plt.close(figv)
    