import matplotlib.pyplot as plt
import numpy as np

def plot_avgstd(x, y, axis=None, color=None, ax=None, fmt="o", label=None,
        alpha=None, figsize=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    avg = np.average(y, axis)
    std = np.std(y, axis)

    ax.errorbar(x, avg, yerr=std, fmt=fmt, color=color, label=label, alpha=alpha)
    return ax

def plot_dub_avgstd(x, y, axis=None, color=None, ax=None, fmt="o"):
    if ax==None:
        fig, ax = plt.subplots()
    yavg = np.average(y, axis) 
    ystd = np.std(y, axis)
    xavg = np.average(x, axis) 
    xstd = np.std(x, axis)

    ax.errorbar(xavg, yavg, yerr=ystd, xerr=xstd, fmt=fmt, color=color)
    return ax

def plot_trial_sem(avg, sem, pre=50, colors=None, labels='None', ax=None,
        alphas=None, figsize=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    for idx in range(len(avg)):
        y = avg[idx]
        yhat = sem[idx]
        x = np.arange(len(y)) - pre
        if labels is not None:
            label = labels[idx]
        else:
            label=None
        if colors is not None:
            color = colors[idx]
        else:
            color=None
        if alphas is not None:
            alpha=alphas[idx]
        else:
            alpha=None
        ax.plot(x, y, color=color, label=label, alpha=alpha)
        ax.fill_between(x, y - yhat, y+yhat, color=color, alpha=0.2)
    ax.axvline(0, linestyle='--', color='tab:blue')
    if labels != 'None':
        ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_raster_co(session, loader='CO', laser=True, ctrl=True):
    fig, ax = plt.subplots()
    if loader == 'CU':
        num_samples = len(session.region1['train'])
        num_r1 = session.num_region1
        num_r2 = session.num_region2
        reg1 = session.region1['train']
        reg2 = session.region2['train']
        temp = np.arange(num_samples)
        region1=[]
        region2=[]
        for idx in np.arange(num_r1):
            region1.append(temp[reg1[:, idx]>0])
        for idx in np.arange(num_r2):
            region2.append(temp[reg2[:,idx]>0])
    else:
        region1 = session.region1['train']
        region2 = session.region2['train']
    trains = [*region1, *region2]
    color1 = ['tab:purple']*len(region1)
    color2 = ['tab:orange']*len(region2)
    colors = [*color1, *color2]
    idx_region2 = np.arange(len(region1), len(region1)+len(region2))
    events = [np.squeeze(train) for train in trains]
    laser_bounds = session.laser['laser_bounds']
    ctrl_bounds = session.laser['ctrl_bounds']
    ax.eventplot(events, color=colors)
    y_max = len(region1) + len(region2)
    if laser:
        for bound in laser_bounds:
            ax.fill_between(x=[bound[0], bound[1]], y1=y_max, y2=len(region1), color='tab:blue', zorder=20)
    if ctrl:
        for bound in ctrl_bounds:
            ax.fill_between(x=[bound[0], bound[1]], y1=y_max, y2=len(region1),
                    color='gray', zorder=30)
    
    return fig, ax
