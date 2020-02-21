# Define some commonly used functions here   \
# Gleb Lukicov (11 Jan 2020)  
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import Series, DataFrame
import sys
import re
from copy import copy
from math import floor, log10

#define common constants
meanS=r"$\mathrm{\mu}$"
sigmaS=r"$\sigma$"
chi2ndfS=r"$\frac{\chi^2}{DoF}$"


def plotHist(data, n_bins=100, prec=2, font_size=14, units="units", input_color="green", alpha=0.7):
    '''
    Input is a 1D array

    # Example of plotting 1D histo from data with automatic binning
    # ax, legend = cu.plotHist(dataX, n_bins=None)
    # cu.textL(ax, 0.8, 0.85, str(legend), font_size=14)
    # plt.show()
    '''
    # 5 DoF stats 
    N, mean, meanE, sd, sdE = stats5(data)
    legend = legend5(N, mean, meanE, sd, sdE, units, prec=prec) # return legend string 

    # seaborn hist plot with input pars
    ax = sns.distplot(data, bins=n_bins, hist=True, kde=False, color=input_color, hist_kws={"alpha": alpha})
   
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel="", fontsize=font_size)
    ax.set_ylabel(ylabel="", fontsize=font_size)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    # plt.tight_layout()

    return ax, legend


def plotHist2D(x, y, n_binsXY=(100, 100), prec=2, font_size=14, units="units", figsize=(10, 10), cmap=plt.cm.jet, cmin=1):
    '''
    Inputs are two 1D arrays

    # Example of getting 2D histro from data 
    jg,cb,legendX,legendY =cu.plotHist2D(dataXY[0], dataXY[1], n_binsXY=(100,100))
    jg.ax_joint.set_ylabel(r"$\theta_y$", fontsize=18)
    jg.ax_joint.set_xlabel(r"$t^{mod}_{g-2}$", fontsize=18)
    plt.savefig("thetavsT.png", dpi=300)

    '''
    # 4DoF stats in X and Y 
    Nx, meanx, meanEx, sdx, sdEx = stats5(x)
    legendX = legend4(meanx, meanEx, sdx, sdEx, units, prec=prec) # return legend string x
    Ny, meany, meanEy, sdy, sdEy = stats5(y)
    legendY = legend4(meany, meanEy, sdy, sdEy, units, prec=prec) # return legend string y

    # the return in JointGrid (not axes)
    # fig : jg.fig, axes : jg.ax_joint
    jg = sns.jointplot(x=x, y=y)
    jg.fig.set_size_inches(figsize[0], figsize[1])
    jg.ax_joint.cla() # clear 
    plt.sca(jg.ax_joint) # join 
    plt.hist2d(x, y, bins=(n_binsXY[0], n_binsXY[1]), cmap=cmap, cmin=cmin) #add 2D histo on top

    #add color bar
    cb = plt.colorbar(use_gridspec=False, orientation="vertical", shrink=0.65, anchor=(1.9, 0.3))
    cb.set_label("Frequency", fontsize=font_size)
    cb.ax.tick_params(labelsize=font_size-1)

    #Make pretty plot as default
    jg.ax_joint.tick_params(labelsize=font_size)
    jg.ax_joint.set_ylabel("Y",fontsize=font_size)
    jg.ax_joint.set_xlabel("X",fontsize=font_size)
    jg.ax_joint.tick_params(axis='x', which='both', bottom=True, top=False, direction='inout')
    jg.ax_joint.tick_params(axis='y', which='both', left=True, right=False, direction='inout')
    jg.ax_joint.minorticks_on()

    #make space for the colour bar
    jg.fig.tight_layout(rect=[0.0, 0.0, 0.85, 0.85])

    # axes can be accessed with cb.ax, jt.
    return jg, cb, legendX, legendY

def plotScatter(x, y, font_size=14, input_color="green", figsize=(12,5), label=None, lw=1, tight=True, step=False):
    
    fig, ax = plt.subplots(figsize=figsize)
    if (not step):
        ax.plot(x, y, c=input_color, label=label, lw=lw)
    if (step):
        ax.step(x, y, where="post", c=input_color, llabel=label, lw=lw)
    
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel="", fontsize=font_size)
    ax.set_ylabel(ylabel="", fontsize=font_size)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    if(tight):
        fig.tight_layout()

    return fig, ax

def get_freq_bin_c_from_data(data, bin_w, bin_range):
    '''
    Return binned data bin frequencies and bin centres
    given bin width and range 
    '''
    bin_n = int(round((bin_range[1] - bin_range[0])/bin_w)) 
    freq, bin_edges = np.histogram(data, bins=bin_n, range=bin_range)
    bin_c=np.linspace(bin_edges[0]+bin_w/2, bin_edges[-1]-bin_w/2, len(freq))
    assert( len(freq) == len(bin_c) ==  bin_n)
    return bin_c, freq 

def residuals(x, y, func, pars):
    '''
    Calcualte fit residuals
    '''
    residuals=[]
    for i in range(0, len(x)): 
        r = y[i] - func(x[i], *pars)  
        residuals.append(r)
    return np.array(residuals)

def chi2_ndf(x, y, y_err, func, pars):
    '''
    Calcualte chi2
    '''    
    chi2=0
    for i in range(0, len(x)): 
    # for i in range(0, len(x)):
        r = y[i] - func(x[i], *pars)  
        chi2+=(r)**2/y_err[i]**2
    ndf = len(x) - len(pars)
    return chi2/ndf, chi2, ndf 

def thetaY_unblinded_phase(t, *pars, phi=6.240):    
    
    A_bz  = pars[0]      
    A_edm = pars[1]    
    c     = pars[2]    
    omega = pars[3]
    
    return A_bz * np.cos(omega * t + phi) + A_edm * np.sin(omega * t + phi) + c


def sin_unblinded(t, A, b, c):
    return A * np.sin(b * t)+c

def gauss(x, *p): 
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def textL(ax, x, y, legend, font_size=14, color="green", weight="normal"):
    '''
    return a good formatted plot legend
    '''
    return ax.text(x, y, str(legend),  fontsize=font_size, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color=color, weight=weight)

def legend5(N, mean, meanE, sd, sdE, units, prec=4):
    '''
    form a string from 5 stats inputs with given precision
    '''
    # form raw string with Latex
    legend = "N={0:d}".format(N)+"\n"+str(meanS)+"={0:.{prec}f}({1:d}) ".format(mean, int(round(meanE*10**prec)), prec=prec)+units+"\n"+str(sigmaS)+"={0:.{prec}f}({1:d}) ".format(sd, int(round(sdE*10**prec)), prec=prec)+units
    return legend

def legend4(mean, meanE, sd, sdE, units, prec=4):
    '''
    form a string from 4 stats inputs with given precision
    '''
    # form raw string with Latex
    legend = "  "+str(meanS)+"={0:.{prec}f}({1:d}) ".format(mean, int(round(meanE*10**prec)), prec=prec)+units+str(sigmaS)+"={0:.{prec}f}({1:d})".format(sd, int(round(sdE*10**prec)), prec=prec)+units
    return legend

def legend4_fit(chi2ndf, mean, meanE, sd, sdE, units, prec=2):
    '''
    form a string from 4 stats inputs with given precision
    '''
    # form raw string with Latex
    legend = "  "+str(chi2ndfS)+"={:.{prec}f}\n".format(chi2ndf, prec=2)+str(meanS)+"={0:.{prec}f}({1:d}) ".format(mean, int(round(meanE*10**prec)), prec=prec)+units+"\n"+str(sigmaS)+"={0:.{prec}f}({1:d}) ".format(sd, int(round(sdE*10**prec)), prec=prec)+units
    return legend

def legend1_fit(chi2ndf, prec=2):
    '''
    form a string from 1 stat inputs with given precision
    '''
    # form raw string with Latex
    
    legend = "  "+str(chi2ndfS)+"={0:.{prec}f}\n".format(chi2ndf, prec=prec)
    return legend

def legend_par(legend, parNames, par, par_e, units, prec=2):
    for i, i_name in enumerate(parNames):
        value=i_name+"={0:+.{prec}f}".format(par[i], prec=prec)+"({0:d})".format( int(round(par_e[i]*10**prec)), prec=prec)+" "+units[i]
        legend+=value+"\n"
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

# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Author: sodd 
    https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting

    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def Profile(x, y, ax, nbins=10, xmin=0, xmax=4, mean=False, sd=False, full_y=False, font_size=14, color="green", only_binned=False):
    '''
    # Return both the plot and DF of binned data 
    '''
    df = DataFrame({'x' : x , 'y' : y})

    binedges = xmin + ((xmax-xmin)/nbins) * np.arange(nbins)
    df['bin'] = np.digitize(df['x'],binedges)

    bincenters = xmin + ((xmax-xmin)/nbins)*np.arange(nbins) + ((xmax-xmin)/(2*nbins))
    ProfileFrame = DataFrame({'bincenters' : bincenters, 'N' : df['bin'].value_counts(sort=False)},index=range(1,nbins+1))

    # store all y as array 
    df_array_y = DataFrame(columns=['array_y'])

    bins = ProfileFrame.index.values
    for bin in bins:
        ProfileFrame.loc[bin,'ymean'] = df.loc[df['bin']==bin,'y'].mean()
        # add the entire y data in the x bin without re-binning the y data (Gleb)
        if (full_y):
            df_array_y.at[bin, 'array_y'] = np.array(df.loc[df['bin']==bin,'y'])

        ProfileFrame.loc[bin,'yStandDev'] = df.loc[df['bin']==bin,'y'].std()
        ProfileFrame.loc[bin,'yMeanError'] = ProfileFrame.loc[bin,'yStandDev'] / np.sqrt(ProfileFrame.loc[bin,'N'])

    if (mean):
        df_binned = DataFrame({'bincenters' : ProfileFrame['bincenters'] , 'ymean' : ProfileFrame['ymean'], 'xerr' : (xmax-xmin)/(2*nbins),  'yerr' : ProfileFrame['yMeanError']})
    elif (sd):
        df_binned = DataFrame({'bincenters' : ProfileFrame['bincenters'] , 'ymean' : ProfileFrame['ymean'], 'xerr' : (xmax-xmin)/(2*nbins),  'yerr' : ProfileFrame['yStandDev']})
    elif (full_y):
        df_binned = DataFrame({'bincenters' : ProfileFrame['bincenters'] , 'y' : df_array_y['array_y'], 'xerr' : (xmax-xmin)/(2*nbins),  'yerr' : ProfileFrame['yMeanError']})
    else:
        raise Exception("Specify either 'mean' or 'sd' y_error as 'True'")
    
    #reset index to start at 0...
    df_binned.reset_index(inplace=True)

    if(not only_binned):
        # make a nice looking plot as default 
        ax.set_xlabel(xlabel="X", fontsize=font_size)
        ax.set_ylabel(ylabel="Y", fontsize=font_size)
        ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
        ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
        ax.minorticks_on()
        plt.xticks(fontsize=font_size-1)
        plt.yticks(fontsize=font_size-1)
        if (mean):
            ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'], yerr=ProfileFrame['yMeanError'], xerr=(xmax-xmin)/(2*nbins), linewidth=0, elinewidth=2, color=color, marker="o") 
        elif (sd):
            ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'], yerr=ProfileFrame['yStandDev'], xerr=(xmax-xmin)/(2*nbins), linewidth=0, elinewidth=2, color=color, marker="o") 
        return ax, df_binned, df
    if(only_binned):
        return df_binned


def profilePlotEqBin(x,y,xmin,xmax,bs,debug=0):
    '''
    Author: Prof. M. Lancaster 
    An alternative implementation of a Profile histogram
    '''
    if (debug == 1):
        print(xmin,xmax,bs)

    ii = np.argsort(x)
    x = x[ii]
    y = y[ii]
    bins = np.arange(xmin-bs/2.0, xmax+bs/2.0+0.1, bs)
    binc = np.arange(xmin,xmax+0.1,bs)
    nbins = len(bins)-1
    inds = np.digitize(x, bins, right=False)

    if (debug == 1):
        print(binc)
        print(x)
        print(y)
        print(inds)

    mean  = np.array([])
    rms   = np.array([])
    emean = np.array([])
    xv = np.array([])
    for i in range(nbins):
        itemindex = np.where(inds==i+1)
        n = itemindex[0].size
        if (n > 5):
            i1 = itemindex[0][0]
            i2 = itemindex[0][-1]

            mean  = np.append(mean,np.mean(y[i1:i2+1]))
            rmsV  = np.nanstd(y[i1:i2+1])
            rms   = np.append(rms, rmsV)
            emean = np.append(emean, rmsV/np.sqrt(i2-i1+1))
            xv    = np.append(xv,binc[i])

    return xv,mean,rms,emean


#no data is returned by sns.regplot, just a pretty plot...really, seaborn?! 
# use Profile instead, plus seaborn is quite slow... 
def plotProfileSNS(x, y, x_estimator=np.mean, bins=10, fit_bool=False, ci=95, marker="+", color="green", font_size=14):
    ''' 
    return axes with a profile plot 
    '''
    fig, ax = plt.subplots(1,1)
    ax = sns.regplot(x=x, y=y, x_estimator=x_estimator, x_bins=bins, fit_reg=fit_bool, marker=marker, color=color, ax=ax)
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel="X", fontsize=font_size)
    ax.set_ylabel(ylabel="Y", fontsize=font_size)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    return ax 