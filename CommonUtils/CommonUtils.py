# Define some commonly used functions here 
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plotHist(data, n_bins=100, prec=4, font_size=14, input_color="green", x_label=r"x [$\mathrm{\mu}$m]", y_label=r"y [$\mathrm{\mu}$m]"):
    '''
    Input is a 1D array 
    '''
    # seaborn with input pars
    ax = sns.distplot(data, bins=n_bins, hist=True, kde=False, color=input_color)

    # 5 DoF stats 
    N, mean, meanE, sd, sdE = stats5(data)
    legend = legend5(N, mean, meanE, sd, sdE, prec) # return legend string 
   
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel=x_label, fontsize=font_size)
    ax.set_ylabel(ylabel=y_label, fontsize=font_size)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    plt.tight_layout()

    return plt, ax, legend


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

