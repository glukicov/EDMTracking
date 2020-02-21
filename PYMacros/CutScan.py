# Scan over mom cuts from TTrees data (streamed into Pandas)
# Gleb Lukicov (20 Feb 2020)  
import os, sys
sys.path.append('../CommonUtils/') # some commonly used functions 
import CommonUtils as cu
import matplotlib as mpl
mpl.use('Agg') # batch mode 
import matplotlib.pyplot as plt

import argparse 
from scipy import stats, optimize # fitting functions 
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import root_numpy # see http://scikit-hep.org/root_numpy/install.html  
from root_pandas import read_root # see https://github.com/scikit-hep/root_pandas 

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--cut", type=str, required=True, help="none, edm, bz")
args=arg_parser.parse_args()

### Constants
omegaAMagic = 0.00143934 # from gm2geom consts / kHz (used to modulate fill time only)

### Define cuts based on input arguments 
if(args.cut == "none"):
    cuts_lower=([0])   
    cuts_upper=([3100])        
elif(args.cut == "edm"):
    cuts_lower=np.linspace(1000, 100, 10, dtype=int)
    cuts_upper=np.linspace(2100, 3000, 10, dtype=int)
elif(args.cut == "bz"):
    cuts_lower=np.linspace(1500, 2000, 6,  dtype=int)
    cuts_upper=3100*np.ones(len(cuts_lower), dtype=int)
else:
    raise Exception("Wrong cut specified!")

#can't use "<" in filename -> "_"
# so form list of string for labelling, and one for filenames 
cutName=[]
cutLabel=[]
for i_len in range(len(cuts_lower)):
    name = str(cuts_lower[i_len])+"<p<"+str(cuts_upper[i_len])
    cutName.append(name)
    cutLabel.append(name.replace("<","_"))

print("Fitting over:", *cutName, "MeV")

## Read in the ROOT Trees into a pandas data frame
data_all = read_root("../DATA/Trees/sim_VLEDM.root", 'trackerNTup/tracker')

### Calculate vetex momentum (as a new column)
data_all['decayVertexMom'] = np.sqrt(data_all["decayVertexMomX"]**2 + data_all["decayVertexMomY"]**2 + data_all["decayVertexMomZ"]**2)


N_array = []
A_edm_array = []
A_bz_array = [] 
chi2_array = [] 
# loop over iterator(0, N) and two cuts (low, high) in parallel 
for i, (low, high) in enumerate(zip(cuts_lower, cuts_upper)):

    print("Starting on cut #", i+1, "out of", len(cuts_lower))

    #form the cut
    mom_cut = (data_all['decayVertexMom'] > low) & (data_all['decayVertexMom'] < high)

    ### From all data select only what we need, based on the cuts 
    data = data_all[mom_cut].loc[:, ["decayVertexMom", "decayVertexMomY", "trackT0", ]]
    N = data.shape[0] # count vertices after cuts 
    data["trackT0"]=data["trackT0"]*1e-3 ### ns - > us 

    # Now get $\theta_y = atan2(\frac{p_y}{p})$ (in mrad) 
    # and modulated $g-2$ time (in us, already done above)
    theta_y = np.arctan2(-data["decayVertexMomY"], data["decayVertexMom"]) *1e3 # mrad 
    g2period = (2*np.pi / omegaAMagic ) * 1e-3 # us 
    g2fracTimeReco = data["trackT0"] / g2period
    g2fracTimeRecoInt = g2fracTimeReco.astype(int)
    modulog2TimeReco = (g2fracTimeReco - g2fracTimeRecoInt) * g2period

    # resolve data into easy variable names 
    time, theta, p= modulog2TimeReco, theta_y, data["decayVertexMom"]

    #sanity plot the momentum cut itself
    ax, legend = cu.plotHist(p, n_bins=45)
    plt.savefig("../fig/cuts/mom_"+cutLabel[i]+".png")
    plt.clf()

    ### Profile (bin), fit, plot
    # return binned dataframe only (only_binned=True)
    df_binned =cu.Profile(time, theta, None, nbins=15, xmin=np.min(time), xmax=np.max(time), mean=True, only_binned=True)

    # resole data for fit 
    x=df_binned['bincenters']
    y=df_binned['ymean']
    x_err=df_binned['xerr']
    y_err=df_binned['yerr']

    '''
    Now perfrom a 4-parameter fit in simulation with the constants phase of 6.240(8) rad 
    $$\theta(t) = A_{B_z}\cos(\omega t + \phi) + A_{\mathrm{EDM}}\sin(\omega t + \phi) + c$$
    where  
    [0] $A_{\mathrm{B_z}}$ is the $B_z$ amplitude   
    [1] $A_{\mathrm{EDM}}$ is the EDM amplitude  
    [2] $c$ is the offset in the central angle  
    [3] $\omega$ is the anomalous precision frequency  (unblinded, for now)
    '''
    par, pcov = optimize.curve_fit(cu.thetaY_unblinded_phase, x, y, sigma=y_err, p0=[0.0, 0.17, 0.0, 1.4], absolute_sigma=False, method='lm')
    par_e = np.sqrt(np.diag(pcov))
    chi2_ndf, chi2, ndf=cu.chi2_ndf(x, y, y_err, cu.thetaY_unblinded_phase, par)
    
    print("Params:", par)
    print("Errors:", par_e)
    print( r"Fit $\frac{\chi^2}{\rm{DoF}}$="+str(round(chi2_ndf,2)) )

    ### Plot
    fig, ax = plt.subplots()
    ax.errorbar(x,y,xerr=x_err, yerr=y_err, linewidth=0, elinewidth=2, color="green", marker="o", label="Data")
    ax.plot(x, cu.thetaY_unblinded_phase(x, *par), color="red", label='Fit')
    ax.legend(loc='best')
    ax.set_ylabel(r"$\langle\theta_y\rangle$ [mrad]", fontsize=16)
    ax.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=16)

    # deal with fitter parameters
    parNames=[r"$ A_{B_z}$", r"$ A_{\rm{EDM}}$", "c", r"$\omega$"]
    units=["mrad", "mrad", "mrad", "MhZ"]
    prec=2
    #form complex legends 
    legend1_chi2=cu.legend1_fit(chi2_ndf)
    legned1_par=""
    legned1_par=cu.legend_par(legned1_par,  parNames, par, par_e, units)
    legend1=legend1_chi2+"\n"+legned1_par
    print(legend1)
    legend2=cutName[i]+"\n N="+cu.sci_notation(N)

    #place on the plot and save 
    y1,y2,x1,x2=0.2,0.85,0.25,0.70
    cu.textL(ax, x1, y1, legend1, font_size=13, color="red")    
    cu.textL(ax, x2, y2, legend2, font_size=14)
    ax.legend(loc='center right', fontsize=16)
    plt.tight_layout() 
    plt.savefig("../fig/cuts/profile_"+cutLabel[i]+".png", dpi=300)
    plt.clf()

    #append for final plots
    N_array.append(N)
    A_bz_array.append(par[0])
    A_edm_array.append(par[1])
    chi2_array.append(chi2_ndf)

#save results to PD 
d = {'Cut':cutName, 'N':N_array,'A_bz':A_bz_array,"A_edm":A_edm_array, "chi2":chi2_array}
df = pd.DataFrame(d)
df.to_csv("../DATA/misc/df_cuts_"+args.cut+".csv")


  



