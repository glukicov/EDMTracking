# Author: Gleb Lukicov (21 Feb 2020)
# A front-end module to run over fitWithBlinders_skim.py iteratively  

import sys, re, os, subprocess, datetime, glob
import argparse
import pandas as pd
import numpy as np
np.set_printoptions(precision=3) # 3 sig.fig 
sys.path.append('../CommonUtils/') # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
from scipy import stats, optimize, fftpack
import matplotlib as mpl
mpl.use('Agg') # MPL in batch mode
font_size=16
import matplotlib.pyplot as plt
import seaborn as sn

#Input fitting parameters 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--all", action='store_true', default=False) # just run all 4 DS 
arg_parser.add_argument("--start", action='store_true', default=False) 
arg_parser.add_argument("--end", action='store_true', default=False) 
arg_parser.add_argument("--plot", action='store_true', default=False) 
arg_parser.add_argument("--plot_end", action='store_true', default=False) 
arg_parser.add_argument("--corr", action='store_true', default=False) 
args=arg_parser.parse_args()

### Constants 
DS_path = (["../DATA/HDF/Sim/Sim.h5"])
stations=([1218])
dss = (["Sim"])
keys=["count", "theta", "truth"]

# DS_path = ("../DATA/HDF/EDM/60h.h5", "../DATA/HDF/EDM/9D.h5", "../DATA/HDF/EDM/HK.h5", "../DATA/HDF/EDM/EG.h5")
# stations=(12, 18)
# dss = (["60h"]) 
# keys=["count", "theta"] # HDF5 keys of input scan result files 

#plotting and retrieving from HDF5 
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N$", r"$\tau$", r"$A$", r"$\phi$"];
par_names_theta= ["A_Bz", "A_edm_blind", "c"]; par_labels_theta= [r"$A_{B_{z}}$", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$", r"$c$"]; 
par_names_theta_truth=par_names_theta.copy(); par_names_theta_truth[1]="A_edm"; par_labels_truth=par_labels_theta.copy(); par_labels_truth[1]=r"$A_{\mathrm{EDM}}$"
par_labels=[par_labels_count, par_labels_theta, par_labels_truth]
par_names=[par_names_count, par_names_theta, par_names_theta_truth] 


bin_w = 10*1e-3 # 150 ns
factor=10
step=bin_w*factor
start=0
stop_desired=5 # us 
dt = stop_desired - start
n_dt = int(dt/bin_w/factor) # how many whole (factor x bins) do we fit in that interval
print("Will generate", n_dt+1, "start times between", start, "and", stop_desired, 'us')
stop = n_dt*factor*bin_w+start
print("Adjusted last start time ",stop)

start_times = np.arange(start, stop, step, dtype=float)
end_times = np.linspace(100, 300, 40, dtype=float)

start_times=[4.4, 5.5, 6.6]
end_times=[100, 150, 200, 250, 300]

print(start_times)
print(end_times)

in_=input("Start scans?")


def main():
    '''
    As a quick solution use sub process 
    '''
    if(args.all): all(DS_path)
    if(args.start): time_scan(DS_path, start_times)
    if(args.end): time_scan(DS_path, end_times)
    if(args.plot): plot(direction="start")
    if(args.plot_end): plot(direction="stop")
    if(args.corr): corr()


def all(DS_path):
    for path in DS_path:
        subprocess.Popen(["python3", "getLongitudinalField.py", "--hdf", path])

def time_scan(DS_path, times):
    subprocess.call(["trash"] + glob.glob("../DATA/scans/edm_scan*"))
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans/*.png") )
    if (args.start==True): key_scan = "--t_min"
    if (args.end==True): key_scan = "--t_max"
    for path in DS_path:
        for time in times:
             subprocess.call(["python3", "getLongitudinalField.py", "--hdf", path, "--scan", key_scan, str(time)])


def plot(direction="start"):
    print("Making scan summary plot")
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans_fom/*.png") )
    
    for i_key, key in enumerate(keys):

        print("Opening", key)

        #the HDF5 has 3 keys 
        data = pd.read_csv("../DATA/scans/edm_scan_"+key+".csv")
        
        par_n=-1
        if(data.shape[1] == 17):  par_n=4 # count 
        if(data.shape[1] == 15):  par_n=3 # theta, truth 
        print("par_n =", par_n, "according to expected total columns")
        if(par_n!=len(par_names[i_key])): raise Exception("More parameters in scan data then expected names - expand!")
        if(par_n==--1): raise Exception("Scan data has less/more then expected parameters!")

        #resolve chi2 - a special parameter and add to data
        chi2 = data['chi2']
        chi2_e=np.sqrt(2/ (data['ndf']-par_n) ) #number of bins - number of fit parameters          
        data['chi2_e']=chi2_e # add chi2_e to the dataframe
        par_names[i_key].insert(0, "chi2")
        par_labels[i_key].insert(0, r"$\frac{\chi^2}{\rm{DoF}}$")


        # print(par_names[0])
        # print(len(par_names[0]))
        # sys.exit()

        for ds in dss:
            for station in stations:
                # apply cuts and select data
                print("S",station,":")
                station_cut = (data["station"]==station)
                ds_cut =  (data['ds']==ds)
                plot_data=data[station_cut & ds_cut]
                plot_data=plot_data.reset_index()

                if(np.max(plot_data.isnull().sum()) > 0): raise Exception("NaNs detected, make method to treat those...")

                # resolve paramters for plotting 
                x=plot_data[direction]

                #loop over all parameters 
                for i_par in range(len(par_names[i_key])):
                    print(par_names[i_key][i_par])
                    y=plot_data[par_names[i_key][i_par]] 
                    y_e=plot_data[par_names[i_key][i_par]+'_e'] 
                    y_s = np.sqrt(y_e**2-y_e[0]**2) # 1sigma band
                    if(args.plot_end): y_s = np.sqrt(y_e[0]**2-y_e**2); # 1sigma band

                    if(y_s.isnull().sum()>0): 
                        print("Error at later times are smaller - not physical, check for bad fit at these times [us]:")
                        print( [x[i] for i, B in enumerate(y_s.isnull()) if B]) # y_s.isnull() list of T/F , B is on of T/F in that list, i is index of True, x[i] times of True
                    
                    #Plot 
                    fig, ax = cu.plot(x, y, y_err=y_e, error=True, elw=2, label="S"+str(station)+": "+ds+" DS", tight=True)
                    ax.plot(x, y, marker=".", ms=10, c="g", lw=0)
                    sigma_index=0; band_P=y[sigma_index]+y_s; band_M=y[sigma_index]-y_s;
                    if(args.plot_end): sigma_index=len(y)-1; band_P=y[sigma_index]-np.flip(y_s); band_M=y[sigma_index]+np.flip(y_s)
                    ax.plot(x, band_P, c="r", ls=":", lw=2, label=r"$\sigma_{\Delta_{21}}$")
                    ax.plot(x, band_M, c="r", ls=":", lw=2)
                    if(par_names[i_key][i_par]=='tau'): ax.plot([np.min(x)-2, np.max(x)+2], [64.44, 64.44], c="k", ls="--"); ax.set_ylim(np.min(y)-0.1, 64.6);
                    if(par_names[i_key][i_par]=='chi2'): ax.plot([min(x)-2, max(x)+2], [1, 1], c="k", ls="--");
                    ax.set_xlim(min(x)-2, max(x)+2)
                    ax.set_xlabel(direction+r" time [$\rm{\mu}$s]", fontsize=font_size);
                    ax.set_ylabel(par_labels[i_key][i_par], fontsize=font_size);
                    ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, 1.1))
                    fig.subplots_adjust(left=0.15)
                    fig.savefig("../fig/scans_fom/"+direction+"_"+key+"_"+par_names[i_key][i_par]+"_S"+str(station)+"_"+str(ds)+".png", dpi=300);

                    # look into parameters
                    #if(par_names[i_key][i_par]=='A_cbo'): print(y, y_e, y_s); sys.exit()
            
def corr():
    '''
    plot correlation matrix for the fit parameters
    '''
    print("Plotting EDM correlations")
    for i_key, key in enumerate(keys):
        for i_station, station in enumerate(stations):
            corr=np.corrcoef(np.load("../DATA/misc/pcov_"+key+"_S"+str(station)+".np.npy"))
            df_corr = pd.DataFrame(corr, columns=par_labels[i_key], index=par_labels[i_key])
            fig,ax = plt.subplots()
            ax=sn.heatmap(df_corr, annot=True, fmt='.2f', linewidths=.5, cmap="bwr")
            cu.textL(ax, 0.5, 1.1, "S"+str(station)+": "+key)
            fig.savefig("../fig/corr_"+key+"_S"+str(station)+".png", dpi=300)


if __name__ == "__main__":
    main()