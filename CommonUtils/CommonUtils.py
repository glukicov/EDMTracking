# Define some commonly used functions here   \
# Gleb Lukicov (11 Jan 2020)  
from scipy import stats, optimize
import numpy as np
import seaborn as sns
# MPL in batch mode
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
import sys, os
import re
from math import floor, log10

# Import blinding libraries 
sys.path.append("../Blinding") # path to Blinding libs
from BlindersPy3 import Blinders
from BlindersPy3 import FitType
getBlinded = Blinders(FitType.Omega_a, "EDM all day") 

#fix the random seed
def get_random_engine(init_seed=123456789):
    return np.random.RandomState(seed=init_seed)

#set printout precision of arrays
np.set_printoptions(precision=9)

#Set constant phase (e.g. cu._phi=x)
_phi=-1
_omega=-1
_LT=-1
_DS=-1

#define common constants
meanS=r"$\mathrm{\mu}$"
sigmaS=r"$\sigma$"
chi2ndfS=r"$\frac{\chi^2}{DoF}$"


def plotHist(data, n_bins=100, prec=2, fs=14, units="units", c="green", alpha=0.7):
    '''
    Input is a 1D array

    # Example of plotting 1D histo from data with automatic binning
    # ax, legend = cu.plotHist(dataX, n_bins=None)
    # cu.textL(ax, 0.8, 0.85, str(legend), fs=14)
    # plt.show()
    '''
    # 5 DoF stats 
    N, mean, meanE, sd, sdE = stats5(data)
    legend = legend5(N, mean, meanE, sd, sdE, units, prec=prec) # return legend string 

    # seaborn hist plot with input pars
    ax = sns.distplot(data, bins=n_bins, hist=True, kde=False, color=c, hist_kws={"alpha": alpha})
   
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel="", fontsize=fs)
    ax.set_ylabel(ylabel="", fontsize=fs)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=fs-1)
    plt.yticks(fontsize=fs-1)
    # plt.tight_layout()

    return ax, legend


def plotHist2D(x, y, n_binsXY=(100, 100), prec=2, fs=14, units="units", figsize=(10, 10), cmap=plt.cm.jet, cmin=1):
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
    cb.set_label("Frequency", fontsize=fs)
    cb.ax.tick_params(labelsize=fs-1)

    #Make pretty plot as default
    jg.ax_joint.tick_params(labelsize=fs)
    jg.ax_joint.set_ylabel("Y",fontsize=fs)
    jg.ax_joint.set_xlabel("X",fontsize=fs)
    jg.ax_joint.tick_params(axis='x', which='both', bottom=True, top=False, direction='inout')
    jg.ax_joint.tick_params(axis='y', which='both', left=True, right=False, direction='inout')
    jg.ax_joint.minorticks_on()

    #make space for the colour bar
    jg.fig.tight_layout(rect=[0.0, 0.0, 0.85, 0.85])

    # axes can be accessed with cb.ax, jt.
    return jg, cb, legendX, legendY

def plot(x, y, x_err=None, y_err=None, fs=14, c="green", 
         figsize=(7,5), label=None, lw=1, elw=2, lc='g', ls="-", tight=False, 
         step=False, scatter=False, error=False, plot=False,
         xlabel=None, ylabel=None):
    '''
    Return nicely-formatted axes and figures
    returns empty (formatted) axes if nothing is provided (default)
    '''
    
    fig, ax = plt.subplots(figsize=figsize)
    if (step):
        ax.step(x, y, where="post", c=c, label=label, lw=lw)
    elif (scatter):
        ax.scatter(x, y, c=c, label=label, lw=lw, ls=ls)
    elif (error):
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, linewidth=0, elinewidth=elw, color=c, marker=None, label=label)
    elif (plot):
        ax.plot(x, y, c=c, label=label, lw=lw, ls=ls)
    else:
        print("No plot style specified, returning a nicely formatted axis only: use it e.g. 'ax.plot()'")
    

    # make a nice looking plot as default 
    ax.set_xlabel(xlabel=xlabel, fontsize=fs)
    ax.set_ylabel(ylabel=ylabel, fontsize=fs)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=fs-1)
    plt.yticks(fontsize=fs-1)
    if(tight):
        # fig.tight_layout()
        #make space for the colour bar
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    return fig, ax


def modulo_wiggle_fit_plot(x, y, func, par, par_e, chi2_ndf, t_mod, t_max, t_min, binW, N,
                                prec=3, # set custom precision 
                                show_cbo_terms=False, 
                                show_loss_terms=False, 
                                data_bool=True,
                                legend_fit=r'Fit: $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]$',
                                legend_data="Run-1 tracker data",
                                key="Quality Tracks",
                                fs=15):
    '''
    Fit and plot folded (modulo wiggle function)
    '''
    
    fig, ax = plt.subplots(figsize=(8, 5))

    #log the y and set axis scales 
    plt.yscale("log")
    ax.set_ylim(min(y)*0.90, max(y)*150)
    ax.set_xlim(0, t_mod)
    label_data="Data: \n"+key+"\n"
    plot_name="_data"
    if (not data_bool):
        ax.set_xlim(0.2, t_mod) 
        ax.set_ylim(70, 4.0e4)
        label_data="Sim.: \n All Tracks \n"
        legend_data="Sim. tracker data"
        plot_name="_sim"

    #split into folds/section for modulo plot 
    left=0 
    i_section = 0

    #loop over folds and plot each section
    while left < t_max:
        right=left + t_mod
        mod_filter = (x >= left) & (x<= right) # boolean mask
        #plot data as step-hist. and the fit function
        #plot for that section and label only the 1st
        ax.step(x=x[mod_filter]-left, y=y[mod_filter], where="post", color="g", label=legend_data if i_section==0 else '')
        ax.plot(x[mod_filter]-left, func( x[mod_filter], *par ) , color="red", label=legend_fit if i_section==0 else '', linestyle=":")
        left=right # get the next fold 
        i_section += 1

    #Put legend and pars values 
    N_str=sci_notation(N)
    textL(ax, 0.82, 0.78, label_data+ r"$p$"+" > 1.8 GeV \n"+str(round(t_min,1))+r" $\rm{\mu}$s < t < "+str(round(t_max,1))+r" $\rm{\mu}$s"+"\n N="+N_str, fs=fs-2,  weight="normal")
    # deal with fitted parameters (to display nicely)
    parNames=[r"$N$", r"$\tau$", r"$A$", r"$R$", r"$\phi$"]
    units=["", r"$\rm{\mu}$s", "", "ppm",  "rad"]
    legned_par=r"$\frac{\chi^2}{\rm{DoF}}$="+str(round(chi2_ndf,2))+"\n"
    legned_par=legend_par(legned_par,  parNames, par, par_e, units, prec=prec)
    textL(ax, 0.17, 0.73, "Fit: "+legned_par, fs=fs-2, c="red", weight="normal")
    if(show_cbo_terms):
        parNames=[r"$\rm{A_{CBO}}$", r"$\omega_{\rm{CBO}}$", r"$\phi_{\rm{CBO}}$", r"$\rm{\tau_{CBO}}$"]
        units=[" ", r"$\rm{\mu}$s", "rad", r"$\rm{\mu}$s"]
        legned_cbo=legend_par("",  parNames, par[5:], par_e[5:], units, prec=prec)
        textL(ax, 0.48, 0.75, r"CBO, $C(t)$"+":\n "+legned_cbo, fs=fs-3, c="red", weight="normal")
    if(show_loss_terms):
        legned_par=legend_1par("", r"$K_{\rm{LM}}$", par[9], par_e[9], " ", prec=prec)
        textL(ax, 0.73, 0.04, "Fit: "+legned_par, fs=fs-2, c="red", weight="normal")

    #axis labels and ticks
    plt.ylabel(r"Counts ($N$) per "+str(int(binW*1e3))+" ns", fontsize=fs)
    plt.xlabel(r"Time [$\mathrm{\mu}$s] (modulo "+str(t_mod)+r" $\mathrm{\mu}$s)", fontsize=fs)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=fs-1)
    plt.yticks(fontsize=fs-1)

    return fig, ax


def get_freq_bin_c_from_data(data, bin_w, bin_range):
    '''
    Return binned data bin frequencies and bin centres
    given bin width and range 
    '''
    bin_n = int(round((bin_range[1] - bin_range[0])/bin_w)) 
    freq, bin_edges = np.histogram(data, bins=bin_n, range=bin_range)
    #print("First bin edge:", bin_edges[0], "last bin edge", bin_edges[-1], "bin width:",  bin_edges[1]- bin_edges[0], "bin n:", bin_n)
    bin_c=np.linspace(bin_edges[0]+bin_w/2, bin_edges[-1]-bin_w/2, len(freq))
    assert( len(freq) == len(bin_c) ==  bin_n)
    return bin_c, freq

def get_g2_mod_time(times):
    '''
    modulate time data by the gm2 period
    '''
    omega_a_magic=_omega
    if(omega_a_magic==-1): raise Exception("Set omega_a via cu._omega=x")

    g2_period=np.pi*2/omega_a_magic # 2pi/rad/s -> 1/Mhz
    g2_frac_time = times / g2_period
    g2_frac_time_int = g2_frac_time.astype(int)
    mod_g2_time = (g2_frac_time - g2_frac_time_int) * g2_period

    return mod_g2_time

def get_edm_mod_times(times):
    '''
    modulate time data by the gm2 period for edm blinding
    '''
    omega_a_magic=_omega 
    phi=_phi
    if (phi == -1): raise Exception("Set constants phase via cu._phi=x")
    if(omega_a_magic==-1): raise Exception("Set omega_a via cu._omega=x")

    g2_period=np.pi*2/omega_a_magic # 2pi/rad/s -> 1/Mhz -> us
    phase_offset = phi / omega_a_magic # us 
    edm_mod_times = np.fmod(times - phase_offset, 2 * g2_period) - g2_period

    return edm_mod_times

def get_edm_weights(edm_mod_times):
    '''
    associated weights to mod times
    '''

    lifetime =_LT
    if (lifetime == -1): raise Exception("Set constants lifetime via cu._LT=x")
    
    weights = np.exp(edm_mod_times / lifetime)
    
    return weights

def get_abs_times_weights(times):
    '''
    Combine the above two functions
    and return the absolute times
    '''
    edm_mod_times = get_edm_mod_times(times)
    weights = get_edm_weights(edm_mod_times)
    
    return np.abs(edm_mod_times), weights


def residuals(x, y, func, pars):
    '''
    Calcualte fit residuals
    '''
    return np.array(y -func(x, *pars))

def chi2_ndf(x, y, y_err, func, pars):
    '''
    Calcualte chi2
    '''    
    ndf = len(x) - len(pars) # total N - fitting pars 
    chi2_i=residuals(x, y, func, pars)**2/y_err**2 # (res**2)/(error**2)
    chi2=chi2_i.sum() # np sum 
    return chi2/ndf, chi2, ndf 


def fit_and_chi2(x, y, y_err, func, p0):
    '''
    fit and calculate chi2
    '''
    # Levenberg-Marquardt algorithm as implemented in MINPACK
    par, pcov = optimize.curve_fit(func, x, y, sigma=y_err, p0=p0, absolute_sigma=False, method='lm')
    par_e = np.sqrt(np.diag(pcov))
    print("Params:", par)
    print("Errors:", par_e)
    chi2ndf, chi2, ndf=chi2_ndf(x, y, y_err, func, par)
    print("𝝌2/dof="+str(round(chi2ndf, 2)) )
    return par, par_e, chi2ndf


def unblinded_wiggle_function(x, *pars):
    '''
    ### Define blinded fit function $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]$,
    where  
    [0] $N$ is the overall normalisation  
    [1] $\tau$ is the boosted muon lifetime $\tau = \gamma \cdot \tau_0 = 29.3\cdot2.2=66.44 \, \mu$s  
    [2] $A$ is the asymmetry  
    [3] $\omega_a$ is the anomalous precision frequency (blinded)  
    [4] $\phi$ is the initial phase  
    '''
    time  = x
    norm  = pars[0]
    life  = pars[1]
    asym  = pars[2]
    omega = pars[3]
    phi   = pars[4]
    
    return norm * np.exp(-time/life) * (1 + asym*np.cos(omega*time + phi))

def unblinded_wiggle_fixed(x, *pars):
    '''
    ### Define blinded fit function $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]$,
    where  
    [0] $N$ is the overall normalisation  
    [1] $\tau$ is the boosted muon lifetime $\tau = \gamma \cdot \tau_0 = 29.3\cdot2.2=66.44 \, \mu$s  
    [2] $A$ is the asymmetry  
    [3] $\omega_a$ is the anomalous precision frequency (blinded)  
    [4] $\phi$ is the initial phase  
    '''

    omega=_omega
    if(omega==-1): raise Exception("Set omega_a via cu._omega=x")

    time  = x
    norm  = pars[0]
    life  = pars[1]
    asym  = pars[2]
    phi   = pars[3]

    
    return norm * np.exp(-time/life) * (1 + asym*np.cos(omega*time + phi))

def blinded_wiggle_function(x, *pars):
    '''
    ### Define blinded fit function $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]$,
    where  
    [0] $N$ is the overall normalisation  
    [1] $\tau$ is the boosted muon lifetime $\tau = \gamma \cdot \tau_0 = 29.3\cdot2.2=66.44 \, \mu$s  
    [2] $A$ is the asymmetry  
    [3] $\omega_a$ is the anomalous precision frequency (blinded)  
    [4] $\phi$ is the initial phase  
    '''
    time  = x
    norm  = pars[0]
    life  = pars[1]
    asym  = pars[2]
    R     = pars[3]
    phi   = pars[4]
    
    omega = getBlinded.paramToFreq(R)
    
    return norm * np.exp(-time/life) * (1 + asym*np.cos(omega*time + phi))


def blinded_wiggle_function_cbo(x, *pars):
    '''
    same as blinded_wiggle_function 
    + 4 CBO terms: Courtesy of J. Price (DocDB:12933)
    $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]\cdot{C(t)}$
    where
    $C(t) = 1.0 + e^{-t / \rm{T_{CBO}}}\rm{A_{CBO}} \cos(\rm{WV_{CBO}} \cdot t + \phi_{\rm{CBO}})$
    '''
    time  = x

    #now add CBO pars
    A_cbo = pars[5]
    w_cbo = pars[6]
    # while (pars[7] < 0):           par[7] += np.pi()*2.0 
    # while (pars[7] > np.pi()*2.0): par[7] -= np.pi()*2.0 
    phi_cbo = pars[7]
    t_cbo = pars[8]
    
    C = 1.0 - ( np.exp(-time / t_cbo) * A_cbo * np.cos(w_cbo * time+ phi_cbo) )
    N = blinded_wiggle_function(time, *pars[0:5])
    return N * C


def blinded_10_par(x, *pars):
    '''
    CBO (4 par) + lost muons (1 par) with constant LT (5 par, 4/5 var 1 const LT)
    '''
    import RUtils as ru

    time  = x
    ds=_DS
    if(ds==-1): raise Exception("DS not set via cu._DS=x")
    K     = pars[-1]

    '''
    The code and the LM spectra courtesy of N. Kinnaird (Muon spin precession frequency extraction and decay positron track fitting in Run-1 of the Fermilab Muon g − 2 experiment, PhD thesis, Boston University (2020).)
    ''' 
    # use a pre-made histograms of muon loss spectrum to get this integral 
    #the times in the histogram are in ns
    L = 1.0 - K  * ru.LM_integral(time*1e3, ds) * -3.75e-7 # the 1e-10 is just an arbitrary scaling factor
    return  blinded_wiggle_function_cbo( time, *pars[0:10] ) * L
  

def thetaY_phase(t, *pars):  
    '''
    \langle \theta(t) \rangle =  A_{\mathrm{B_z}}\cos(\omega t + \phi) + A_{\mathrm{EDM}}\sin(\omega t + \phi) + c
    '''  
    phi=_phi
    if (phi == -1): raise Exception("Set constants phase via cu._phi=x")
    omega=_omega
    if(omega==-1): raise Exception("Set omega_a via cu._omega=x")

    A_bz  = pars[0]      
    A_edm = pars[1]    
    c     = pars[2]    
    
    return A_bz * np.cos(omega * t + phi) + A_edm * np.sin(omega * t + phi) + c


def sin(t, A, b, p, c):
    return A * np.sin(b * t+p)+c

def cos(t, A, b, p, c):
    return A * np.cos(b * t+p)+c

def gauss(x, *p): 
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def textL(ax, x, y, legend, fs=14, c="green", weight="normal"):
    '''
    return a good formatted plot legend
    '''
    return ax.text(x, y, str(legend),  fontsize=fs, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color=c, weight=weight)

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
        if (par_e[i] < 1):
            value=i_name+"={0:+.{prec}f}".format(par[i], prec=prec)+"({0:d})".format( int(round(par_e[i]*10**prec)), prec=prec)+" "+units[i]
        else:
            value=i_name+"={0:d}".format(int(round(par[i])))+"({0:d})".format( int(round(par_e[i])))+" "+units[i] 
        legend+=value+"\n"
    return legend

def legend_1par(legend, parName, par, par_e, units, prec=2):
    if (par_e < 1):
        value=parName+"={0:+.{prec}f}".format(par, prec=prec)+"({0:d})".format( int(round(par_e*10**prec)), prec=prec)+" "+units
    else:
        value=parName+"={0:d}".format(int(round(par)))+"({0:d})".format( int(round(par_e)))+" "+units 
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



def Profile(x, y, w=None, ax=None, nbins=10, xmin=0, xmax=4, mean=False, sd=False, full_y=False, fs=14, c="green", only_binned=False):
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
        ax.set_xlabel(xlabel="X", fontsize=fs)
        ax.set_ylabel(ylabel="Y", fontsize=fs)
        ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
        ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
        ax.minorticks_on()
        plt.xticks(fontsize=fs-1)
        plt.yticks(fontsize=fs-1)
        if (mean):
            ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'], yerr=ProfileFrame['yMeanError'], xerr=(xmax-xmin)/(2*nbins), linewidth=0, elinewidth=2, color=c, marker=None) 
        elif (sd):
            ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'], yerr=ProfileFrame['yStandDev'], xerr=(xmax-xmin)/(2*nbins), linewidth=0, elinewidth=2, color=c, marker=None) 
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
def plotProfileSNS(x, y, x_estimator=np.mean, bins=10, fit_bool=False, ci=95, marker="+", color="green", fs=14):
    ''' 
    return axes with a profile plot 
    '''
    fig, ax = plt.subplots(1,1)
    ax = sns.regplot(x=x, y=y, x_estimator=x_estimator, x_bins=bins, fit_reg=fit_bool, marker=marker, color=color, ax=ax)
    # make a nice looking plot as default 
    ax.set_xlabel(xlabel="X", fontsize=fs)
    ax.set_ylabel(ylabel="Y", fontsize=fs)
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='inout')
    ax.minorticks_on()
    plt.xticks(fontsize=fs-1)
    plt.yticks(fontsize=fs-1)
    return ax 