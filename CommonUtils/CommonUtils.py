# Define some commonly used functions here 
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def plotHist(data, n_bins=100, prec=4, font_size=14, input_color="green", alpha=0.7):
    '''
    Input is a 1D array 
    '''
    # seaborn with input pars
    ax = sns.distplot(data, bins=n_bins, hist=True, kde=False, color=input_color, hist_kws={"alpha": alpha})

    # 5 DoF stats 
    N, mean, meanE, sd, sdE = stats5(data)
    legend = legend5(N, mean, meanE, sd, sdE, prec) # return legend string 
   
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel="", fontsize=font_size)
    ax.set_ylabel(ylabel="", fontsize=font_size)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    plt.tight_layout()

    return ax, legend


def plotHist2D(x, y, n_binsX=100, n_binsY=100, prec=4, font_size=14, cmap=plt.cm.jet, alpha=0.7):
    '''
    Inputs are two 1D arrays
    '''
    #make a seborn plot, with 2D histo on top
    #fig = plt.figure()
    ax = sns.jointplot(x=x, y=y)
    ax.ax_joint.cla()
    plt.sca(ax.ax_joint)
    plt.hist2d(x, y, bins=(100, 100), cmap=cmap);

    # 5 DoF stats 
    N, mean, meanE, sd, sdE = stats5(x)
    legendX = legend5(N, mean, meanE, sd, sdE, prec) # return legend string x
    N, mean, meanE, sd, sdE = stats5(y)
    legendY = legend5(N, mean, meanE, sd, sdE, prec) # return legend string y

    # make a nice looking plot as default 
    plt.ylabel("",fontsize=font_size)
    plt.xlabel("",fontsize=font_size)
    cbaxes = ax.fig.add_axes([0.8, 0.05, 0.05, 0.8]) 
    cb = plt.colorbar(cax = cbaxes)  
    cb.ax.tick_params(labelsize=font_size-1) 
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    # TODO fine tune other decorations 
    # ax.fig.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    # ax.fig.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    # ax.fig.minorticks_on()
    # plt.tight_layout()

    return ax, cb, legendX, legendY 



def textL(ax, x, y, legend, font_size=14):
    '''
    return a good formatted plot legend
    '''
    return ax.text(x, y, str(legend),  fontsize=font_size, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')

def legend5(N, mean, meanE, sd, sdE, prec=4):
    '''
    form a string from 5 stats inputs with given precision
    '''
    legend = "N={0:d} \n Mean={1:.{prec}f}({2:d}) \n SD={3:.{prec}f}({4:d})".format(N, mean, int(round(meanE*10**prec)), sd, int(round(sdE*10**prec)), prec=prec)
    return legend

def stats5(data):
    '''
    Input is a 1D array 
    '''
    N = len(data)
    mean = np.mean(data)
    meanE = stats.sem(data)
    sd = np.std(data)
    sdE = np.sqrt(sd**2/ (2*N) )
    return N, mean, meanE, sd, sdE 

def stats3(data):
    '''
    Input is a 1D array 
    '''
    N = len(data)
    mean = np.mean(data)
    meanE = stats.sem(data)
    return N, mean, meanE

