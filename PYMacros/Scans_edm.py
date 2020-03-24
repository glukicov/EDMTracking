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
arg_parser.add_argument("--p_min", action='store_true', default=False) 
arg_parser.add_argument("--p_both", action='store_true', default=False) 
arg_parser.add_argument("--period", action='store_true', default=False) 
arg_parser.add_argument("--plot_start", action='store_true', default=False) 
arg_parser.add_argument("--plot_end", action='store_true', default=False) 
arg_parser.add_argument("--plot_p_min", action='store_true', default=False) 
arg_parser.add_argument("--plot_p_both", action='store_true', default=False) 
arg_parser.add_argument("--plot_period", action='store_true', default=False) 
arg_parser.add_argument("--corr", action='store_true', default=False) 
args=arg_parser.parse_args()

### Constants 
DS_path = (["../DATA/HDF/Sim/Sim.h5"])
stations=([1218])
dss = (["Sim"])
keys=["count", "theta", "truth"]
key_names=["(count)", r"($\theta$)", "(truth)"]

# DS_path = ("../DATA/HDF/EDM/60h.h5", "../DATA/HDF/EDM/9D.h5", "../DATA/HDF/EDM/HK.h5", "../DATA/HDF/EDM/EG.h5")
# stations=(12, 18)
# dss = (["60h"]) 
# keys=["count", "theta"] # HDF5 keys of input scan result files 

#plotting and retrieving from HDF5 
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N$ (count)", r"$\tau$", r"$A$", r"$\phi$"];
par_names_theta= ["A_Bz", "A_edm_blind", "c"]; par_labels_theta= [r"$A_{B_{z}}$", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$", r"$c$"]; 
par_names_theta_truth=par_names_theta.copy(); par_names_theta_truth[1]="A_edm"; par_labels_truth=par_labels_theta.copy(); par_labels_truth[1]=r"$A_{\mathrm{EDM}}$"
par_labels=[par_labels_count, par_labels_theta, par_labels_truth]
par_names=[par_names_count, par_names_theta, par_names_theta_truth] 


if(args.start): 
    bin_w = 10*1e-3 # 150 ns
    factor=40
    step=bin_w*factor
    start=0
    stop_desired=10 # us 
    dt = stop_desired - start
    n_dt = int(dt/bin_w/factor) # how many whole (factor x bins) do we fit in that interval
    print("Will generate", n_dt+1, "start times between", start, "and", stop_desired, 'us')
    stop = n_dt*factor*bin_w+start
    print("Adjusted last start time ",stop)
    start_times = np.arange(start, stop, step, dtype=float)
    print("Start times:", start_times)

if(args.end):   
    end_times = np.linspace(100, 300, 15, dtype=float)
    print("End times:", end_times)

if(args.p_min):   
    p_min = np.linspace(1000, 2400, 15, dtype=float)
    print("P min:", p_min)

if(args.p_both):
    p_min = np.linspace(0, 1000, 11, dtype=float)
    p_max = np.linspace(3100, 2100, 11, dtype=float)
    print("P min:", p_min)
    print("P max:", p_max)

if(args.period):
    period=np.linspace(4.3, 4.4, 12, dtype=float)
    print("period:", period)

in_=input("Start scans/plots?")


def main():
    '''
    As a quick solution use sub process 
    '''
    if(args.all): all(DS_path)
    if(args.start): time_scan(DS_path, start_times)
    if(args.end): time_scan(DS_path, end_times)
    if(args.p_min): time_scan(DS_path, p_min)

    if(args.p_both): both_scan(DS_path, p_min, p_max)
    
    if(args.plot_start): plot(direction="start")
    if(args.plot_end): plot(direction="stop")
    if(args.plot_p_min): plot(direction="p_min")
    if(args.plot_p_min): plot(direction="")
    if(args.plot_p_both): plot(direction="p_min", bidir=True, second_direction="p_max")
    if(args.corr): corr()


def all(DS_path):
    for path in DS_path:
        subprocess.Popen(["python3", "getLongitudinalField.py", "--hdf", path])

def time_scan(DS_path, times):
    subprocess.call(["trash"] + glob.glob("../DATA/scans/edm_scan*"))
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans/*.png") )
    if (args.start==True): key_scan = "--t_min"
    if (args.end==True): key_scan = "--t_max"
    if (args.p_min==True): key_scan = "--p_min"
    for path in DS_path:
        for time in times:
             subprocess.call(["python3", "getLongitudinalField.py", "--hdf", path, "--scan", key_scan, str(time)])

def both_scan(DS_path, p_min, p_max):
    subprocess.call(["trash"] + glob.glob("../DATA/scans/edm_scan*"))
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans/*.png") )
    for path in DS_path:
        for i, mom in enumerate(p_min):
            subprocess.call(["python3", "getLongitudinalField.py", "--hdf", path, "--scan", "--p_min", str(p_min[i]), "--p_max",  str(p_max[i])])

def plot(direction="start", bidir=False, second_direction=None):
    print("Making scan summary plot")
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans_fom/*.png") )
    
    for i_key, key in enumerate(keys):

        print("Opening", key)

        #the HDF5 has 3 keys 
        data = pd.read_csv("../DATA/scans/edm_scan_"+key+".csv")
        
        par_n=-1
        if(data.shape[1] == 18):  par_n=3 # theta 
        if(data.shape[1] == 17):  par_n=4 # count 
        if(data.shape[1] == 15):  par_n=3 # truth 
        print("par_n =", par_n, "according to expected total columns")
        if(par_n!=len(par_names[i_key])): raise Exception("More parameters in scan data then expected names - expand!")
        if(par_n==--1): raise Exception("Scan data has less/more then expected parameters!")

        #resolve chi2 - a special parameter and add to data
        chi2 = data['chi2']
        chi2_e=np.sqrt(2/ (data['ndf']-par_n) ) #number of bins - number of fit parameters          
        data['chi2_e']=chi2_e # add chi2_e to the dataframe
        par_names[i_key].insert(0, "chi2")
        par_labels[i_key].insert(0, r"$\frac{\chi^2}{\rm{DoF}}$")

        #resolve NA parameters
        par_names[i_key].insert(0, "n")
        par_labels[i_key].insert(0, "N (stat)")
        data['n_e']=np.zeros(data.shape[0]) # add no error on the number of entries

        #resolve A for each key 
        if(key=="count"):
            data['NA2']=data['n']*(data['A']**2)
            par_names[i_key].insert(0, "NA2")
            par_labels[i_key].insert(0, r"NA$^2$")
            data['NA2_e']=np.zeros(data.shape[0]) # add no error on the number of entries

        if(key=="theta"):
            data["NA_bz2"]=data['n']*(data['A_Bz']**2)
            par_names[i_key].insert(0, "NA_bz2")
            par_labels[i_key].insert(0, r"NA$_{B_{z}}^2$")
            data['NA_bz2_e']=np.zeros(data.shape[0]) # add no error on the number of entries
            
            data["NA_edm2"]=data['n']*(data['A_edm_blind']**2)
            par_names[i_key].insert(0, "NA_edm2")
            par_labels[i_key].insert(0, r"NA$_{\rm{EDM}}^2$")
            data['NA_edm2_e']=np.zeros(data.shape[0]) # add no error on the number of entries


        if(key=="truth"):
            data["NA_bz2"]=data['n']*(data['A_Bz']**2)
            par_names[i_key].insert(0, "NA_bz2")
            par_labels[i_key].insert(0, r"NA$_{B_{z}}^2$")
            data['NA_bz2_e']=np.zeros(data.shape[0]) # add no error on the number of entries
            
            
            data["NA_edm2"]=data['n']*(data['A_edm']**2)
            par_names[i_key].insert(0, "NA_edm2")
            par_labels[i_key].insert(0, r"NA$_{\rm{EDM}}^2$")
            data['NA_edm2_e']=np.zeros(data.shape[0]) # add no error on the number of entries



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

                if(bidir==True):
                    x1=plot_data[direction]; x2=plot_data[second_direction]
                    x=[str(int(x1[i]))+"-"+str(int(x2[i])) for i, s in enumerate(x1)]        

                #loop over all parameters 
                for i_par in range(len(par_names[i_key])):
                    print(par_names[i_key][i_par])
                    y=plot_data[par_names[i_key][i_par]] 
                    y_e=plot_data[par_names[i_key][i_par]+'_e'] 
                    y_s = np.sqrt(y_e**2-y_e[0]**2) # 1sigma band
                    # y_s = np.sqrt(np.abs(y_e**2-y_e[0]**2)) # 1sigma band
                    if(args.plot_end): y_s = np.sqrt(y_e[0]**2-y_e**2); # 1sigma band

                    if(y_s.isnull().sum()>0): 
                        print("Error at later times are smaller - not physical, check for bad fit at these times [us]:")
                        print( [x[i] for i, B in enumerate(y_s.isnull()) if B]) # y_s.isnull() list of T/F , B is on of T/F in that list, i is index of True, x[i] times of True
                    
                    #Plot 
                    fig, ax = cu.plot(x, y, y_err=y_e, error=True, elw=2, label="S"+str(station)+": "+ds+" DS "+key_names[i_key], tight=False)
                    ax.plot(x, y, marker=".", ms=10, c="g", lw=0)
                    sigma_index=0; band_P=y[sigma_index]+y_s; band_M=y[sigma_index]-y_s;
                    if(args.plot_end): sigma_index=len(y)-1; band_P=y[sigma_index]-np.flip(y_s); band_M=y[sigma_index]+np.flip(y_s)
                    ax.plot(x, band_P, c="r", ls=":", lw=2, label=r"$\sigma_{\Delta_{21}}$")
                    ax.plot(x, band_M, c="r", ls=":", lw=2)
                    # if(par_names[i_key][i_par]=='tau'): ax.plot([np.min(x)-2, np.max(x)+2], [64.44, 64.44], c="k", ls="--"); ax.set_ylim(np.min(y)-0.1, 64.6);
                    if(par_names[i_key][i_par]=='chi2' and not args.plot_p_both): ax.plot([min(x)-2, max(x)+2], [1, 1], c="k", ls="--");
                    if(not args.plot_p_both): ax.set_xlim(min(x)-2, max(x)+2);
                    ax.set_xlabel(direction+r" time [$\rm{\mu}$s]", fontsize=font_size);
                    if(args.plot_p_min): ax.set_xlabel(r"$p_{\rm{min}}$ [MeV]", fontsize=font_size);
                    if(args.plot_p_both): 
                        ax.set_xlabel(r"$p$ [MeV]", fontsize=font_size) 
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(45)
                    ax.set_ylabel(par_labels[i_key][i_par], fontsize=font_size);
                    # ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, 1.1))
                    ax.legend(fontsize=font_size, loc="best")
                    fig.savefig("../fig/scans_fom/"+direction+"_"+key+"_"+par_names[i_key][i_par]+"_S"+str(station)+"_"+str(ds)+".png", dpi=300, bbox_inches='tight');

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