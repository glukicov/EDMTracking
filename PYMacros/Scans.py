# Author: Gleb Lukicov (21 Feb 2020)
# A front-end module to run over fitWithBlinders_skim.py iteratively  

import sys, re, subprocess
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
arg_parser.add_argument("--corr", action='store_true', default=False) 
args=arg_parser.parse_args()

### Constants 
# DS_path = ("../DATA/HDF/MMA/60h.h5", "../DATA/HDF/MMA/9D.h5", "../DATA/HDF/MMA/HK.h5", "../DATA/HDF/MMA/EG.h5")
DS_path = (["../DATA/HDF/MMA/60h.h5"])
start_times = np.linspace(30.2876, 100, 36, dtype=float)
end_times = np.linspace(400, 500, 36, dtype=float)

# print(start_times)
# sys.exit()
# print(end_times)

stations=(12, 18)
dss = ("60h", "9D", "HK", "EG")

def main():
    '''
    As a quick solution use sub process 
    TODO: import and run module properly
    '''

    if(args.all): all(DS_path)

    if(args.start): time_scan(DS_path, start_times)
    if(args.end): time_scan(DS_path, end_times)
    if(args.plot): plot()
    if(args.corr): corr()


def all(DS_path):
    for path in DS_path:
        # subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path])
        subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--cbo"])
        # subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--loss"])

def time_scan(DS_path, times):
    if (args.start==True): key = "--min"
    if (args.end==True): key = "--max"
    for path in DS_path:
        for time in times:
            # subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--loss", "--scan", key, str(time)])
            subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--cbo", "--scan", key, str(time)])


def plot():
    data = pd.read_csv("../DATA/misc/scans/scan.csv")
    par_n=-1
    if(data.shape[1] == 25):  par_n=10
    if(data.shape[1] == 25):  par_n=9
    if(data.shape[1] == 17):  par_n=5
    print("par_n =", par_n, "according to expected total columns")

    for ds in dss:
        for station in stations:
            # apply cuts and select data
            station_cut = (data["station"]==station)
            ds_cut =  (data['ds']==ds)
            plot_data= data[station_cut & ds_cut]
            plot_data=plot_data.reset_index()

            # resolve paramters for plotting 
            start=plot_data['start']
            stop=plot_data['stop']
            
            chi2 = plot_data['chi2']
            chi2_e=np.sqrt(2/ (plot_data['ndf']-par_n) )

            x=start
            y=chi2
            y_e=chi2_e 
            y_s = np.sqrt(y_e**2-y_e[0]**2) # 1sigma band


def corr():
    '''
    plot correlation matrix for the fit parameters
    '''
    corr=[np.corrcoef(np.load("../DATA/misc/pcov_S12.np.npy")), np.corrcoef(np.load("../DATA/misc/pcov_S18.np.npy"))]
    names=[r"$N$", r"$\tau$", r"$A$", r"$R$", r"$\phi$", r"$A_{\rm{CBO}}$", r"$\omega_{\rm{CBO}}$", r"$\phi_{\rm{CBO}}$", r"$\tau_{\rm{CBO}}$"]
    for i_station, station in enumerate(stations):
        df_corr = pd.DataFrame(corr[i_station],columns=names, index=names)
        fig,ax = plt.subplots()
        ax=sn.heatmap(df_corr, annot=True, fmt='.2f', linewidths=.5, cmap="bwr")
        cu.textL(ax, 0.5, 1.1, "S"+str(station))
        fig.savefig("../fig/corr_S"+str(station)+".png", dpi=300)


if __name__ == "__main__":
    main()