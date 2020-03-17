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
arg_parser.add_argument("--edm", action='store_true', default=False) 
args=arg_parser.parse_args()

### Constants 
# DS_path = ("../DATA/HDF/MMA/60h.h5", "../DATA/HDF/MMA/9D.h5", "../DATA/HDF/MMA/HK.h5", "../DATA/HDF/MMA/EG.h5")
DS_path = (["../DATA/HDF/MMA/60h.h5"])


bin_w = 149.2*1e-3 # 150 ns
factor=10
step=bin_w*factor
stop_desired=90 # us 
start=30.2876 # us 
dt = stop_desired - start
n_dt = int(dt/bin_w/factor) # how many whole (factor x bins) do we fit in that interval
print("Will generate", n_dt+1, "start times between", start, "and", stop_desired, 'us')
stop = n_dt*factor*bin_w+start
print("Adjusted last start time ",stop)

start_times = np.arange(start, stop, step, dtype=float)
end_times = np.linspace(300, 500, 36, dtype=float)

# print(start_times)
# sys.exit()
# print(end_times)

stations=(12, 18)
# dss = ("60h", "9D", "HK", "EG")
dss = (["60h"])
par_names=["N", "tau", "A", "R", "phi", "A_cbo", "w_cbo", "phi_cbo", "tau_cbo", "K_LM",]
# par_names=["N", "tau", "A", "R", "phi", "A_cbo", "w_cbo", "phi_cbo", "tau_cbo"]
par_labels=[r"$N$", r"$\tau$", r"$A$", r"$R$", r"$\phi$", r"$A_{\rm{CBO}}$", r"$\omega_{\rm{CBO}}$", r"$\phi_{\rm{CBO}}$", r"$\tau_{\rm{CBO}}$", r"$K_{\rm{LM}}$",]
# par_labels=[r"$N$", r"$\tau$", r"$A$", r"$R$", r"$\phi$", r"$A_{\rm{CBO}}$", r"$\omega_{\rm{CBO}}$", r"$\phi_{\rm{CBO}}$", r"$\tau_{\rm{CBO}}$"])

def main():
    '''
    As a quick solution use sub process 
    TODO: import and run module properly
    '''
    if(args.all): all(DS_path)
    if(args.start): time_scan(DS_path, start_times)
    if(args.end): time_scan(DS_path, end_times)
    if(args.plot): plot(direction="start")
    if(args.plot_end): plot(direction="stop")
    if(args.corr): corr()


def all(DS_path):
    for path in DS_path:
        # subprocess.Popen(["python3", "fitWithBlinders_skim.py", "--hdf", path])
        # subprocess.Popen(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--cbo"])
        subprocess.Popen(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--loss"])
        # subprocess.Popen(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--loss", "--min", "55.6516"])

def time_scan(DS_path, times):
    subprocess.call(["mv", "../DATA/scans/scan.csv", "../DATA/scans/scan_"+str(int(datetime.datetime.now().timestamp()))+".csv"]) # backup previous file
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans/*.png") )
    if (args.start==True): key = "--min"
    if (args.end==True): key = "--max"
    for path in DS_path:
        for time in times:
            # subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--cbo", "--scan", key, str(time)])
            subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--loss", "--scan", key, str(time)])


def plot(direction="start"):
    print("Making scan summary plot")
    subprocess.Popen( ["trash"] + glob.glob("../fig/scans_fom/*.png") )
    data = pd.read_csv("../DATA/scans/scan.csv")
    par_n=-1
    if(data.shape[1] == 27):  par_n=10
    if(data.shape[1] == 25):  par_n=9 
    if(data.shape[1] == 17):  par_n=5
    print("par_n =", par_n, "according to expected total columns")
    if(par_n!=len(par_names)): raise Exception("More parameters in scan data then expected names - expand!")
    if(par_n==--1): raise Exception("Scan data has less/more then expected parameters!")

    #resolve chi2 - a special parameter and add to data
    chi2 = data['chi2']
    chi2_e=np.sqrt(2/ (data['ndf']-par_n) ) #number of bins - number of fit parameters          
    data['chi2_e']=chi2_e # add chi2_e to the dataframe
    par_names.insert(0, "chi2")
    par_labels.insert(0, r"$\frac{\chi^2}{\rm{DoF}}$")

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
            for i_par in range(par_n):
                print(par_names[i_par])
                y=plot_data[par_names[i_par]] 
                y_e=plot_data[par_names[i_par]+'_e'] 
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
                if(par_names[i_par]=='tau'): ax.plot([np.min(x)-2, np.max(x)+2], [64.44, 64.44], c="k", ls="--"); ax.set_ylim(np.min(y)-0.1, 64.6);
                if(par_names[i_par]=='chi2'): ax.plot([min(x)-2, max(x)+2], [1, 1], c="k", ls="--");
                ax.set_xlim(min(x)-2, max(x)+2)
                ax.set_xlabel(direction+r" time [$\rm{\mu}$s]", fontsize=font_size);
                ax.set_ylabel(par_labels[i_par], fontsize=font_size);
                ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, 1.1))
                fig.subplots_adjust(left=0.15)
                fig.savefig("../fig/scans_fom/"+direction+"_"+par_names[i_par]+"_S"+str(station)+"_"+str(ds)+".png", dpi=300);

                # look into parameters
                #if(par_names[i_par]=='A_cbo'): print(y, y_e, y_s); sys.exit()
            
def corr():
    '''
    plot correlation matrix for the fit parameters
    '''
    if(not args.edm):
        corr=[np.corrcoef(np.load("../DATA/misc/pcov_S12.np.npy")), np.corrcoef(np.load("../DATA/misc/pcov_S18.np.npy"))]
        for i_station, station in enumerate(stations):
            df_corr = pd.DataFrame(corr[i_station],columns=par_labels, index=par_labels)
            fig,ax = plt.subplots()
            ax=sn.heatmap(df_corr, annot=True, fmt='.2f', linewidths=.5, cmap="bwr")
            cu.textL(ax, 0.5, 1.1, "S"+str(station))
            fig.savefig("../fig/corr_S"+str(station)+".png", dpi=300)
    if(args.edm):
        par_labels=[r"$N$", r"$\tau$", r"$A$", r"$\phi$"]
        corr=np.corrcoef(np.load("../DATA/misc/pcov_N.np.npy"))
        df_corr = pd.DataFrame(corr, columns=par_labels, index=par_labels)
        fig,ax = plt.subplots()
        ax=sn.heatmap(df_corr, annot=True, fmt='.2f', linewidths=.5, cmap="bwr")
        # cu.textL(ax, 0.5, 1.1, "S"+str(station))
        fig.savefig("../fig/corr_N.png", dpi=300)

        par_labels=[r"$A_{B_{z}}$", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$", r"$c$"]
        corr=np.corrcoef(np.load("../DATA/misc/pcov_theta.np.npy"))
        df_corr = pd.DataFrame(corr, columns=par_labels, index=par_labels)
        fig,ax = plt.subplots()
        ax=sn.heatmap(df_corr, annot=True, fmt='.2f', linewidths=.5, cmap="bwr")
        # cu.textL(ax, 0.5, 1.1, "S"+str(station))
        fig.savefig("../fig/corr_theta.png", dpi=300)





if __name__ == "__main__":
    main()