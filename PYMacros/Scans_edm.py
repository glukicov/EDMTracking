
# Author: Gleb Lukicov (21 Feb 2020)
# A front-end module to run over fitWithBlinders_skim.py iteratively  

import sys, re, os, subprocess, datetime, glob
import time as timeMod
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
import matplotlib.patches as patches
import matplotlib.ticker as plticker
import seaborn as sn

#Input fitting parameters 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--all", action='store_true', default=False) # just run all 4 DS 
arg_parser.add_argument("--mom", action='store_true', default=False)  
arg_parser.add_argument("--all_mom", action='store_true', default=False)  
arg_parser.add_argument("--plot_mom", action='store_true', default=False)  
arg_parser.add_argument("--plot_all_mom", action='store_true', default=False)  
arg_parser.add_argument("--plot_bz", action='store_true', default=False)  

arg_parser.add_argument("--start", action='store_true', default=False) 
arg_parser.add_argument("--stop", action='store_true', default=False) 
arg_parser.add_argument("--p_min", action='store_true', default=False) 
arg_parser.add_argument("--p_minp_max", action='store_true', default=False) 
arg_parser.add_argument("--g2period", action='store_true', default=False) 
arg_parser.add_argument("--phase", action='store_true', default=False) 
arg_parser.add_argument("--lt", action='store_true', default=False) 
arg_parser.add_argument("--bin_w", action='store_true', default=False) 

arg_parser.add_argument("--plot_start", action='store_true', default=False) 
arg_parser.add_argument("--plot_stop", action='store_true', default=False) 
arg_parser.add_argument("--plot_p_min", action='store_true', default=False) 
arg_parser.add_argument("--plot_p_minp_max", action='store_true', default=False) 
arg_parser.add_argument("--plot_g2period", action='store_true', default=False) 
arg_parser.add_argument("--plot_phase", action='store_true', default=False) 
arg_parser.add_argument("--plot_lt", action='store_true', default=False) 
arg_parser.add_argument("--plot_bin_w", action='store_true', default=False) 

arg_parser.add_argument("--equal", action='store_true', default=False) 
arg_parser.add_argument("--corr", action='store_true', default=False) 
arg_parser.add_argument("--band_off", action='store_true', default=False) 
arg_parser.add_argument("--abs", action='store_true', default=False) 
arg_parser.add_argument("--input", type=str, default=None) 

arg_parser.add_argument("--both", action='store_true', default=False, help="Separate fists for S12 and S18")
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/EDM/R1.h5", help="Full path to the data file: HDF5")
arg_parser.add_argument("--file", type=str, default="../DATA/scans/edm_scan_theta.csv", help="Full path to the data file: HDF5")
args=arg_parser.parse_args()

### Constants 
key_names=["(count)", r"($\theta$)", "(truth)"]
g2period  = round(1/0.2290735,6) 

stations=(12, 18)
if (args.both):stations=([1218])

expected_DSs = ("60h", "9D", "HK",   "EG", "Sim", "Bz", "noBz", "R1")
official_DSs = ("Run-1a",  "Run-1c",  "Run-1b",  "Run-1d",  "Sim",  r"$B_z$ = 1700 ppm", "noBz",  "Run-1")
if(not args.plot_mom):
    DS_path=([args.hdf])
    ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
    ds_name_official=official_DSs[expected_DSs.index(ds_name)]
    folder=args.hdf.replace(".","/").split("/")[-3] 
    print("Detected DS name:", ds_name, ds_name_official, "from the input file!")
    if( (folder != "Sim") and (folder != "EDM")): raise Exception("Load a pre-skimmed simulation or an EDM file")
    if(not ds_name in expected_DSs): raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")
    cu._DS=ds_name
    dss=([ds_name])
    label1=ds_name_official
    if (dss[0]=='Bz'):
        args.both=True
        stations=([1218])

# keys=["count", "theta"] # HDF5 keys of input scan result files 
##plotting and retrieving from HDF5 
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N$ (count)", r"$\tau$"+r"[$\rm{\mu}$s]", r"$A$", r"$\phi$ [rad]"];
par_names_theta= ["A_Bz"]; par_labels_theta= [r"$A_{B_{z}}$"+r" [$\rm{\mu}$rad]"]; 
# par_names_theta= ["A_Bz", "A_edm_blind", "c"]; par_labels_theta= [r"$A_{B_{z}}$ [mrad]", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$ [mrad]", r"$c$ [mrad]"]; 
# par_names_theta_truth=par_names_theta.copy(); par_names_theta_truth[1]="A_edm"; par_labels_truth=par_labels_theta.copy(); par_labels_truth[1]=r"$A_{\mathrm{EDM}}$"
# par_labels=[par_labels_count, par_labels_theta, par_labels_truth]
# par_names=[par_names_count, par_names_theta, par_names_theta_truth] 

keys=["theta"] # HDF5 keys of input scan result files 
par_labels=[par_labels_theta]
par_names=[par_names_theta] 

# keys=["count"] # HDF5 keys of input scan result files s
# par_labels=[par_labels_count]
# par_names=[par_names_count] 

# par_names_count= ["A"];    par_labels_count= [r"$N$ (count)"];
# par_names_theta= ["A_Bz"]; par_labels_theta= [r"$A_{B_{z}}$ [mrad]"]; 
# par_labels=[par_labels_count, par_labels_theta]; par_names=[par_names_count, par_names_theta]; 
# keys=["count", "theta"] # HDF5 keys of input scan result files 

# par_names_theta= ["A_Bz"]; par_labels_theta= [r"$A_{B_{z}}$ ["+r"$\rm{\mu}$rad]"]; 
# par_labels=[par_labels_theta]; par_names=[par_names_theta]; 
# keys=["theta"] # HDF5 keys of input scan result files 

if(args.start): 
    factor=1
    step=g2period*factor
    start=30.56
    stop_desired=80 # us 
    dt = stop_desired - start
    n_dt = int(dt/g2period/factor) # how many whole (factor x bins) do we fit in that interval
    print("Will generate", n_dt+1, "start times between", start, "and", stop_desired, 'us')
    stop = n_dt*factor*g2period+start
    print("Adjusted last start time ",stop)
    start_times = np.arange(start, stop, step, dtype=float)
    print("Start times:", start_times)
    in_=input("Start scans?")

if(args.stop):   
    # end_times = np.linspace(300, 450, 9, dtype=float)
    factor=3
    step=g2period*factor
    start=323.04
    stop_desired=480 # us 
    dt = stop_desired - start
    n_dt = int(dt/g2period/factor) # how many whole (factor x bins) do we fit in that interval
    print("Will generate", n_dt+1, "start times between", start, "and", stop_desired, 'us')
    stop = n_dt*factor*g2period+start
    print("Adjusted last start time ",stop)
    end_times = np.arange(start, stop, step, dtype=float)
    print("End times:", end_times)
    in_=input("Start scans?")

if(args.p_min):   
    p_min = np.linspace(800, 2300, 16, dtype=float)
    print("P min:", p_min)
    in_=input("Start scans?")

if(args.mom):

    if(dss[0]=="Bz"):
        p_min = [800,  1200, 1500, 1800]
        p_max = [1200, 1500, 1800, 2300]
    else:   
        # p_min = [800,  1200, 1500, 1800]
        # p_max = [1200, 1500, 1800, 2300]
        #p_min = np.linspace(1000,  2400, 8, dtype=float)
        #p_max = np.linspace(1200,  2600, 8, dtype=float)
        p_min = np.linspace(900,   2400, 16, dtype=float)
        p_max = np.linspace(1000,  2500, 16, dtype=float)

    # p_min = np.linspace(1000,  2200,  7, dtype=float)
    # p_max = np.linspace(1200,  2400,  7, dtype=float)

    # p_min = np.linspace(1600,  2000, 5, dtype=float)
    # p_max = np.linspace(1700,  2100, 5, dtype=float)

    # p_min = [900,  1000, 2500, 2600]
    # p_max = [1000, 1100, 2600, 2700]

    print("P min:", p_min)
    print("P max:", p_max)
    in_=input("Start scans?")

if(args.p_minp_max):
    # p_min = np.linspace(0, 1400, 15, dtype=float)
    # p_max = np.linspace(3100, 1700, 15, dtype=float)

    p_min = np.linspace(500,  1400, 10, dtype=float)
    p_max = np.linspace(2500, 1600, 10, dtype=float)

    # p_min = np.linspace(1000, 2400, 8, dtype=float)
    # p_max = np.linspace(1200, 2600, 8, dtype=float)

    # p_min = ([800])
    # p_max = ([1000])

    # p_min = np.linspace(500, 2500, 9, dtype=float)
    # p_max = np.linspace(700, 2700, 9, dtype=float)


    print("P min:", p_min)
    print("P max:", p_max)
    in_=input("Start scans?")

if(args.g2period): 
    # period=np.linspace(4.2, 4.4, 11, dtype=float)
    period=np.linspace(g2period*(1-30e-6), g2period*(1+30e-6), 11, dtype=float)
    print("period:", period)
    in_=input("Start scans?")

if(args.phase): 
    phase=np.linspace(2.060, 2.090, 11, dtype=float)
    print("phase:", phase)
    in_=input("Start scans?")

if(args.lt): 
    lt=np.linspace(56, 68, 11, dtype=float)
    print("lifetime:", lt)
    in_=input("Start scans?")

if(args.bin_w): 
    bins=np.linspace(5, 400, 16, dtype=float)
    # bins = (5, 15, 30, 50, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400)
    print("bins width [ns]:", bins)
    in_=input("Start scans?")


def main():
    '''
    As a quick solution use sub process 
    '''
    if(args.all): all(["../DATA/HDF/EDM/60h.h5", "../DATA/HDF/EDM/9D.h5", "../DATA/HDF/EDM/HK.h5", "../DATA/HDF/EDM/EG.h5"])
    if(args.all_mom): all_mom(["../DATA/HDF/EDM/60h.h5", "../DATA/HDF/EDM/9D.h5", "../DATA/HDF/EDM/HK.h5", "../DATA/HDF/EDM/EG.h5"])
    if(args.start): time_scan(DS_path, start_times, "--t_min")
    if(args.stop): time_scan(DS_path, end_times, "--t_max")
    if(args.p_min): time_scan(DS_path, p_min, "--p_min", eL="--p_max=3100")
    if(args.g2period): time_scan(DS_path, period, "--g2period")
    if(args.phase): time_scan(DS_path, phase, "--phase")
    if(args.lt): time_scan(DS_path, lt, "--lt")
    if(args.bin_w): time_scan(DS_path, bins, "--bin_w")

    if(args.p_minp_max): both_scan(DS_path, p_min, p_max)
    if(args.mom):        both_scan(DS_path, p_min, p_max)
    
    if(args.plot_start): plot(direction="start")
    if(args.plot_stop): plot(direction="stop")
    if(args.plot_p_min): plot(direction="p_min")
    if(args.plot_g2period): plot(direction="g2period")
    if(args.plot_phase): plot(direction="phase")
    if(args.plot_lt): plot(direction="lt")
    if(args.plot_bin_w): plot(direction="bin_w")
    if(args.plot_p_minp_max): plot(direction="p_min", bidir=True, second_direction="p_max")
    if(args.plot_mom): plot_mom()
    if(args.plot_bzz): 
        paths=["../DATA/scans/60h_edm_scan_theta.csv", "../DATA/scans/HK_edm_scan_theta.csv", "../DATA/scans/9D_edm_scan_theta.csv", "../DATA/scans/EG_edm_scan_theta.csv"]
        for path in paths:        
            plot_mom(df=path, scan=True)
    
    if(args.plot_all_mom): plot_all_mom()

    if(args.corr): corr()

  


def all(DS_path):
    for path in DS_path:
        subprocess.Popen(["python3", "getLongitudinalField.py", "--hdf", path])

def all_mom(DS_path):
    for path in DS_path:
        subprocess.call(["python3", "getLongitudinalField.py", "--hdf", path, "--scan"])
        subprocess.call(["python3", "getLongitudinalField.py", "--hdf", path, "--scan", "--both"])

def time_scan(DS_path, times, key_scan):
    # subprocess.call(["trash"] + glob.glob("../DATA/scans/edm_scan*"))
    #subprocess.Popen( ["trash"] + glob.glob("../fig/scans/*.png") )
    for path in DS_path:
        for time in times:
             # subprocess.call(["python3", "Fast_getLongitudinalField.py", "--hdf", path, "--scan", key_scan, str(time)])
             subprocess.call(["python3", "getLongitudinalField.py", "--hdf", path, "--scan", key_scan, str(time)])

def both_scan(DS_path, p_min, p_max):
    # subprocess.call(["trash"] + glob.glob("../DATA/scans/edm_scan*"))
    #subprocess.Popen( ["trash"] + glob.glob("../fig/scans/*.png") )
    for path in DS_path:
        for i, mom in enumerate(p_min):
            subprocess.call(["python3",  "getLongitudinalField.py", "--hdf", path, "--scan", "--p_min", str(p_min[i]), "--p_max",  str(p_max[i])])

def plot(direction="start", bidir=False, second_direction=None):
    print("Making scan summary plot")
    # subprocess.Popen( ["trash"] + glob.glob("../fig/scans_fom/*.png") )
    

    for i_key, key in enumerate(keys):

        print("Opening", key)
        if( (args.plot_lt or args.plot_phase) and key!="theta"): continue # skip counts of doing LT or phase

        #the HDF5 has 3 keys 
        if(args.input == None): data = pd.read_csv("../DATA/scans/edm_scan_"+key+".csv")
        else: data = pd.read_csv(args.input)
        
        par_n=-1
        if(data.shape[1] == 23):  par_n=3 # theta 
        if(data.shape[1] == 20):  par_n=4 # count 
        if(data.shape[1] == 18):  par_n=3 # truth 
        print("par_n =", par_n, "according to expected total columns")
        #if(par_n!=len(par_names[i_key])): raise Exception("More parameters in scan data then expected names - expand!")
        if(par_n==-1): raise Exception("Scan data has less/more then expected parameters!")

        #resolve chi2 - a special parameter and add to data
        # chi2 = data['chi2']
        # # print(chi2)
        # chi2_e=np.sqrt(2/ (data['ndf']-par_n) ) #number of bins - number of fit parameters
        # # print(chi2_e)          
        # data['chi2_e']=chi2_e # add chi2_e to the dataframe
        # par_names[i_key].insert(0, "chi2")
        # par_labels[i_key].insert(0, r"$\frac{\chi^2}{\rm{DoF}}$")

        # #resolve NA parameters
        # par_names[i_key].insert(0, "n")
        # par_labels[i_key].insert(0, "N (stat)")
        # data['n_e']=np.zeros(data.shape[0]) # add no error on the number of entries

        # # resolve A for each key 
        # if(key=="count"):
        #     data['NA2']=data['n']*(data['A']**2)
        #     par_names[i_key].insert(0, "NA2")
        #     par_labels[i_key].insert(0, r"NA$^2$")
        #     data['NA2_e']=np.zeros(data.shape[0]) # add no error on the number of entries

        # # if(key=="theta"):
        #     data["NA_bz2"]=data['n']*(data['A_Bz']**2)
        #     par_names[i_key].insert(0, "NA_bz2")
        #     par_labels[i_key].insert(0, r"NA$_{B_{z}}^2$")
        #     data['NA_bz2_e']=np.zeros(data.shape[0]) # add no error on the number of entries
            
        #     data["NA_edm2"]=data['n']*(data['A_edm_blind']**2)
        #     par_names[i_key].insert(0, "NA_edm2")
        #     par_labels[i_key].insert(0, r"NA$_{\rm{EDM}}^2$")
        #     data['NA_edm2_e']=np.zeros(data.shape[0]) # add no error on the number of entries


        # if(key=="truth"):
        #     data["NA_bz2"]=data['n']*(data['A_Bz']**2)
        #     par_names[i_key].insert(0, "NA_bz2")
        #     par_labels[i_key].insert(0, r"NA$_{B_{z}}^2$")
        #     data['NA_bz2_e']=np.zeros(data.shape[0]) # add no error on the number of entries
            
            
        #     data["NA_edm2"]=data['n']*(data['A_edm']**2)
        #     par_names[i_key].insert(0, "NA_edm2")
        #     par_labels[i_key].insert(0, r"NA$_{\rm{EDM}}^2$")
        #     data['NA_edm2_e']=np.zeros(data.shape[0]) # add no error on the number of entries


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
                if(args.plot_bin_w):   plot_data=plot_data.sort_values(by=['ndf'],  ascending=False);  plot_data=plot_data.reset_index()
                x=plot_data[direction]
                dirName=direction

                if(bidir):
                    plot_data=plot_data.sort_values(by=['p_min'])
                    plot_data=plot_data.reset_index()
                    x1=plot_data[direction]; x2=plot_data[second_direction]
                    # print(x1)
                    # print(x2)
                    # sys.exit()
                    x=[str(int(x1[i]))+"-"+str(int(x2[i])) for i, s in enumerate(x1)]
                    dirName=direction+second_direction

                
                
                if(args.plot_bin_w): 
                    x=x*1e3 # us to ns     


                #loop over all parameters 
                for i_par in range(len(par_names[i_key])):
                    print(par_names[i_key][i_par])
                    y=plot_data[par_names[i_key][i_par]] 
                    y_e=plot_data[par_names[i_key][i_par]+'_e'] 
                    if(par_names[i_key][i_par]=='A_Bz'):
                        #mrad to urad
                        y=y*1e3
                        y_e=y_e*1e3

                    if(args.band_off): y_s = np.sqrt(y_e**2-y_e[0]**2) # 1sigma band
                    if(args.abs): y_s = np.sqrt(np.abs(y_e**2-y_e[0]**2)) # 1sigma band
                    if(args.plot_stop): y_s = np.sqrt(y_e[0]**2-y_e**2); # 1sigma band
                    if(args.plot_stop and args.abs): y_s = np.sqrt(np.abs(y_e[0]**2-y_e**2)); # 1sigma band

                    # if(y_s.isnull().sum()>0): 
                    #     print("Error at later times are smaller - not physical, check for bad fit at these times [us]:")
                    #     print( [x[i] for i, B in enumerate(y_s.isnull()) if B]) # y_s.isnull() list of T/F , B is on of T/F in that list, i is index of True, x[i] times of True
                    
                    #Plot 
                    fig, ax = cu.plot(x, y, y_err=y_e, error=True, elw=2, label=ds_name_official, tight=False,  marker=".")
                    # ax.plot(x, y, marker=".", ms=10, c="g", lw=0)
                    #sigma_index=0; band_P=y[sigma_index]+y_s; band_M=y[sigma_index]-y_s;
                    if(args.plot_start):
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+7)
                        ax.set_xlim(28, 79)

                    if(args.plot_stop): sigma_index=len(y)-1; band_P=y[sigma_index]-np.flip(y_s); band_M=y[sigma_index]+np.flip(y_s)
                    if(args.band_off): ax.plot(x, band_P, c="r", ls=":", lw=2, label=r"$\sigma_{\Delta_{21}}$"); ax.plot(x, band_M, c="r", ls=":", lw=2)
                    # if(par_names[i_key][i_par]=='tau'): ax.plot([np.min(x)-2, np.max(x)+2], [64.44, 64.44], c="k", ls="--"); ax.set_ylim(np.min(y)-0.1, 64.6);
                    if(par_names[i_key][i_par]=='chi2' and not args.plot_p_minp_max): ax.plot([min(x)-2, max(x)+2], [1, 1], c="k", ls="--");
                    if(args.plot_p_minp_max ): 
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
                        # ax.set_xlim(800,);
                    if(args.plot_p_min ): ax.set_xlim(min(x)-200, max(x)+200); ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
                    ax.set_xlabel(direction+r"-time [$\rm{\mu}$s]", fontsize=font_size);
                    
                    if(args.plot_lt): 
                        ax.set_xlabel(r"$\tau$"+r" [$\rm{\mu}$s]", fontsize=font_size); ax.set_xlim(min(x)*0.95, max(x)*1.05)

                        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.line, (1.0, 1.0) )
                        print(par)
                        ax.plot(x, cu.line(x, *par), c="red", label=r"$\frac{\Delta A_{B_{z}}}{\Delta \tau}$="+str(int(par[0]*1e3)).replace("-", u"\u2212")+r"$\times 10^{-3} \ \rm{\mu}$rad/$\rm{\mu}$s", lw=2)
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+1.2)
                        ax.set_xlim(55, 69)

                    if(args.plot_phase): 
                        ax.set_xlabel(r"$\phi$"+r" [rad]", fontsize=font_size); ax.set_xlim(min(x)*0.999, max(x)*1.001)
                        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.line, (1.0, 1.0) )
                        print(par)
                        ax.plot(x, cu.line(x, *par), c="red", label=r"$\frac{\Delta A_{B_{z}}}{\Delta \phi}$="+str(int(round(par[0],0)))+r"$\ \rm{\mu}$rad/rad", lw=2)
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+1.5)

                    if(args.plot_g2period): 
                        ax.set_xlabel(r"$T_{g-2}$ "+r"[$\rm{\mu}$s]", fontsize=font_size); ax.set_xlim(min(x)*0.999995, max(x)*1.000005); plt.xticks(fontsize=12)
                        x2= ( (plot_data["g2period"]-g2period)/g2period ) * 1e6; 
                        x2=x2.astype(int)
                        ax2 = ax.twiny()
                        ax2.set_xticks(x);
                        ax2.set_xticklabels(x2);
                        ax2.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
                        ax2.set_xlabel(r"$T_{g-2}$ "+r"(ppm)", fontsize=font_size-4);
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+2)

                        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x2, y, y_e, cu.line, (1.0, 1.0) )
                        print(par)
                        ax.plot(x, cu.line(x, *par), c="red", label=r"$\frac{\Delta A_{B_{z}}}{\Delta T_{g-2}}$="+str(int(round(par[0]*1e3)))+r"$\times 10^{-3} \ \rm{\mu}$rad/ppm", lw=2)
                        

                    if(args.plot_bin_w):

                        ax.set_xlabel("Bin width [ns]", fontsize=font_size); 
                        x2=plot_data["ndf"]
                        ax2=ax.twiny()
                        ax2.set_xticks(x[0::2]);
                        ax2.set_xticklabels(x2[0::2]);
                        
                        ax2.set_xlabel("Number of bins", fontsize=font_size-2)

                        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.line, (1.0, 1.0) )
                        print(par)
                        ax.plot(x, cu.line(x, *par), c="red", label=r"$\frac{\Delta A_{B_{z}}}{\Delta \rm{Bin \ width}}$="+str(int(round(par[0]*1e4))).replace("-", u"\u2212")+r"$\times 10^{-4} \ \rm{\mu}$rad/ns", lw=2)

                        # ax.set_xlim(3, 28)
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+2.8)
                        ax2.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])

                    if(args.plot_p_min and par_names[i_key][i_par]=='NA2'):
                        # par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y,  cu.parab, (-1.0, 2400, 1000))
                        par, pcov = optimize.curve_fit(cu.na2, x, y, p0=(-1.0, 1.0, 1.0, 1.0, 1.0), method='lm')
                        print(*par)
                        max_x = optimize.fminbound(lambda x: -cu.na2(x, *par), 1000, 2000) 
                        print(max_x)
                        ax.plot(x, cu.na2(x, *par), c="red", label="Numerical max: "+str(int(max_x))+" MeV", lw=2)

    
                    if(args.plot_p_min): ax.set_xlabel(r"$p_{\rm{min}}$ [MeV]", fontsize=font_size);  ax.set_xlim(min(x)*0.95, max(x)*1.05)
                    if(args.plot_p_minp_max): 
                        ax.set_xlabel(r"$p$ [MeV]", fontsize=font_size) 
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(45)
                    ax.set_ylabel(par_labels[i_key][i_par], fontsize=font_size);
                    # ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, 1.1))
                    
                    if(args.plot_p_min or args.plot_p_minp_max and dss[0]=="Bz"):
                        ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1])
                        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]],[170, 170], c="r", ls=":")

                        x_frac = []
                        for x_i in x:
                            # x_frac.append( (float(x_i.split("-")[1])+float(x_i.split("-")[0])+250)/(2*2800) )
                            x_frac.append( (float(x_i.split("-")[1])+float(x_i.split("-")[0]))/(2*3100) )
                        ax2=ax.twiny()
                        ax2.set_xticks(x_frac);
                        x_frac_round=np.around(x_frac,decimals=2)
                        ax2.set_xticklabels(x_frac_round);
                        ax2.set_xlabel(r"$y=\frac{p}{p_{\rm{max}}}$", labelpad=10)
                        ax2.set_xlim(x_frac[0]-0.05, x_frac[-1]+0.05)


                    if(args.plot_p_min or args.plot_p_minp_max and dss[0]=="noBz"):
                        ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1])
                        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]],[0, 0], c="r", ls=":")

                    
                    if(dss[0] == "Bz"):
                        ax.legend(fontsize=14, loc='lower center')
                        ax.set_ylim(-40, 245) 
                    else: ax.legend(fontsize=font_size, loc="best")
                    fig.savefig("../fig/"+dirName+"_"+key+"_"+par_names[i_key][i_par]+"_S"+str(station)+"_"+str(ds)+".png", dpi=300, bbox_inches='tight');

                    if(args.plot_p_minp_max and dss[0]=="Bz"):
                        print('Calculating asymmetry term')

                        print("Ranges:", x)

                        x_frac = []
                        for x_i in x:
                            # x_frac.append( (float(x_i.split("-")[1])+float(x_i.split("-")[0])+250)/(2*2800) )
                            x_frac.append( (float(x_i.split("-")[1])+float(x_i.split("-")[0]))/(2*3100) )
                        
                        print("Mid point as a fraction of 2800 MeV", x_frac)
     
                        
                        fig, ax = cu.plot(x_frac, y/1700, y_err=y_e/1700, error=True, elw=2, label=ds_name_official, tight=False,  marker=".")
                        
                        print("Asymmetry:", y/1700)

                        ax.set_xticks(x_frac);
                        ax.set_xticklabels(x)
                        ax.set_xlim(x_frac[0]-0.05, x_frac[-1]+0.05)

                        ax2=ax.twiny()
                        ax2.set_xticks(x_frac);
                        x_frac_round=np.around(x_frac,decimals=2)
                        ax2.set_xticklabels(x_frac_round);
                        ax2.set_xlabel(r"$y=\frac{p}{p_{\rm{max}}}$", labelpad=10)
                        ax2.set_xlim(x_frac[0]-0.05, x_frac[-1]+0.05)


                        ax.set_xlabel(r"$p$ [MeV]", fontsize=font_size) 
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(45)
                        ax.set_ylabel(r"Asymmetry-dilution, $d_{B_z}(p)$", fontsize=font_size);
                        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.6)
                        ax.legend(fontsize=14, loc='lower center')
                        
                        
                        y_lin = np.linspace(0, 1, 100) 
                        def A_edm(y_lin):
                            return 0.3*( np.sqrt(y_lin*(1-y_lin))*(1+4*y_lin)  ) / ( 5+5*y_lin-4*y_lin**2  ) 

                        def NA2_edm(y_lin):
                        #     return 60*N(y_lin)*A_edm(y_lin)**2
                            return 0.5*( y_lin*(1-y_lin)**2 * (1+4*y_lin)**2 ) / ( 5 + 5*y_lin - 4*y_lin**2 )
                        # ax.plot(y_lin, A_edm(y_lin), c="b", ls=":", label=r"$A_{\rm{EDM}}(y)=\frac{\sqrt{y(1-y)}(1+4y)}{5+5y-4y^2}$");
                        #ax.plot(y_lin, NA2_edm(y_lin), c="r", ls="--", label=r"$NA^2_{\rm{EDM}}(y)$", lw=2);


                        ax.legend(fontsize=font_size, loc="lower center")
                        ax.set_ylim(-0.015, 0.15) 
                        fig.savefig("../fig/asymm.png", dpi=300, bbox_inches='tight');
                        #fig.savefig("../fig/"+"asymm"+"_"+key+"_"+par_names[i_key][i_par]+"_S"+str(station)+"_"+str(ds)+".png", dpi=300, bbox_inches='tight');


                        #Fit asymmetry
                        x_mid = np.array(x1+x2)/2
                        fig, ax = cu.plot(x_mid, y/1700, y_err=y_e/1700, error=True, elw=2, label=ds_name_official, tight=False,  marker=".")
                        ax.set_ylim(-0.015, 0.15)
                        ax.set_xlim(1000, 2400)
                        ax.set_xlabel(r"$p$ [MeV]", fontsize=font_size);
                        ax.set_ylabel(r"Asymmetry-dilution, $d_{B_z}(p)$", fontsize=font_size);

                        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x_mid, y/1700, y_e/1700, cu.parab, [1,1,1])
                        x_lin = np.linspace(0, 3100, 1000) 
                        ax.plot(x_lin, cu.parab(x_lin, *par), c="r", ls="-", label=r"$d_{B_z}(p)=ap^2+bp+d_0$", lw=2);

                        ax.legend(fontsize=font_size, loc="lower center")
                        fig.savefig("../fig/fit_asymm.png", dpi=300, bbox_inches='tight');


                    # look into parameters
                    #if(par_names[i_key][i_par]=='A_cbo'): print(y, y_e, y_s); sys.exit()

    #when done reading the file - backup
    # subprocess.call(["mv", "../DATA/cans/edm_scan_count.csv", "../DATA/scans/edm_scan_count_"+dirName+".csv"]) # backup previous file
    # subprocess.call(["mv", "../DATA/scans/edm_scan_theta.csv", "../DATA/scans/edm_scan_theta_"+dirName+".csv"]) # backup previous file

def plot_mom(df=args.file, scan=False):
    print("Mom study plots")

    data = pd.read_csv(df)
    data=data.sort_values(by=['p_min'])

    ds_name = data['ds'][0]
    print("Dataset", ds_name)
    dss=( [ ds_name ] )
    ds_name_official=official_DSs[expected_DSs.index(ds_name)]
    label1=ds_name_official

    if(len(stations) == 1):

        station=1218

        p_min = data['p_min']
        p_max = data['p_max']
        cuts = [str(int(p_min[i]))+r"$<p<$"+str(int(p_max[i])) for i,k in enumerate(p_min)] 
        x=np.arange(1,data.shape[0]+1)

        N=data['n']
        print("Fraction of events in each bin\n", np.round(N/np.sum(N),2))
        print("A_Bz in each bin\n", np.round(data['A_Bz']*1e3,2))

        print("N in each bin:", N)
        print("N total:", np.sum(N))

        print("Max N in a bin", max(N))
        print("Min N in a bin", min(N))

        label=label1+" S"+str(station)
        if(args.equal):
            if(not np.all(np.equal(N[0], N))):
                raise Exception("Not all bins have same number of tracks")
            else:
                label=label1+"\n" +r" $N_{\rm{bin}}$="+str(cu.sci_notation(N[0]))

        p_mean = (p_min+p_max)/2

        # plot A_bz        
        fig, ax = cu.plot_mom(x, data['A_Bz']*1e3, data['A_Bz_e']*1e3, cuts, N_s1218=N, p_mean=p_mean, weighted=False, label1=label)
        ax.set_ylabel(r"$A_{B_z} \ [\rm{\mu}$rad]")
        if(dss[0]=='Bz'):
            # plt.legend(fontsize=14, loc=(0.03, 0.66))
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]],[170, 170], c="r", ls=":")
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.3)
        fig.savefig("../fig/sum_mom_A_bz"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        fig, ax = cu.plot_mom(x, data['A_Bz_e']*1e3, None, cuts, N_s1218=N, p_mean=p_mean, weighted=False, label1=label)
        ax.set_ylabel(r"$\delta A_{B_z} \ [\rm{\mu}$rad]")
        #if(dss[0]=='Bz'):
            # plt.legend(fontsize=14, loc=(0.03, 0.66))
            # ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.3)
        fig.savefig("../fig/sum_mom_delta_A_bz"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # # plot A_edm
        # fig, ax = cu.plot_mom(x, data['A_edm_blind']*1e3, data['A_edm_blind_e']*1e3, cuts, N, label2=r"$\langle A_{\rm{EDM}}  \rangle =$", label1=label1+" S"+str(station))
        # ax.set_ylabel(r"$A_{\rm{EDM}} \ [\rm{\mu}$rad]")
        # fig.savefig("../fig/sum_mom_A_edm"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        ##plot c
        fig, ax = cu.plot_mom(x, data['c']*1e3, data['c_e']*1e3, cuts, N_s1218=N, label2=r"$\langle c \rangle =$", label1=label)
        ax.set_ylabel(r"$c \ [\rm{\mu}$rad]")
        fig.savefig("../fig/sum_mom_c"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # #plot sigma 
        fig, ax = cu.plot_mom(x, data['sigma_y'], None, cuts, N_s1218=N, weighted=False, label1=label)
        ax.set_ylabel(r"$\sigma_{\theta_y}$  [mrad]")
        fig.savefig("../fig/sum_mom_sigma"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

    if(len(stations) == 2):

        station=1218

        # s1218=False

        data_s12=data[data['station']==12]
        data_s18=data[data['station']==18]

        data_s12=data_s12.reset_index()
        N=data_s12['n']
        data_s18=data_s18.reset_index()

        p_min = data_s12['p_min']
        p_max = data_s12['p_max']
        cuts = [str(int(p_min[i]))+r"$<p<$"+str(int(p_max[i])) for i,k in enumerate(p_min)] 
        x=np.arange(1,data_s12.shape[0]+1)
        p_mean = (p_min+p_max)/2
        
        n_label=""
        if(args.equal):
            if(not np.all(np.equal(N[0], N))):
                raise Exception("Not all bins have same number of tracks")
            else:
                n_label="\n" +r" $N_{\rm{bin}}$="+str(cu.sci_notation(N[0]))

        # print("Max N in a bin", max(data_s12['n']), max(data_s18['n']))
        # print("Min N in a bin", min(data_s12['n']), min(data_s18['n']))
        # print(N[-1])
        # loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals

        # plot A_bz
        fig, ax = cu.plot_mom(p_min, data_s12['A_Bz']*1e3, data_s12['A_Bz_e']*1e3, p_mean=p_mean, label1=label1+" S12"+n_label, s18=True, s18_y=data_s18['A_Bz']*1e3, s18_y_e=data_s18['A_Bz_e']*1e3, weighted=False)
        # ax.xaxis.set_major_locator(loc)
        ax.set_ylabel(r"$A_{B_z} \ [\rm{\mu}$rad]")
        fig.savefig("../fig/+"ds_name"+sum_mom_A_bz"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        fig, ax = cu.plot(p_mean, cu.get_asym(p_mean), scatter=True)
        # ax.xaxis.set_major_locator(loc)
        ax.set_ylabel(r"$d_{B_z}(p)$")
        ax.set_xlabel(r"$p$ [MeV]")
        fig.savefig("../fig/+"ds_name"+sum_mom_d"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        fig, ax = cu.plot_mom(p_min, data_s12['A_Bz']*1e3, data_s12['A_Bz_e']*1e3, scan=scan, ds_name=ds_name_official, p_mean=p_mean,  label1=label1+" S12"+n_label, s18=True, s18_y=data_s18['A_Bz']*1e3, s18_y_e=data_s18['A_Bz_e']*1e3, weighted=True, asym=True)
        # ax.xaxis.set_major_locator(loc)
        ax.set_ylabel(r"$B_z$ (ppm)")
        fig.savefig("../fig/+"ds_name"+sum_mom_B_z"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # A_bz uncertainty
        # fig, ax = cu.plot_mom(p_min, data_s12['A_Bz_e']*1e3, None, cuts, N, label1=label1+" S12"+n_label, s18=True, s18_y=data_s18['A_Bz_e']*1e3, s18_y_e=None, weighted=False, pmin=True)
        # # ax.xaxis.set_major_locator(loc)
        # ax.set_ylabel(r"$\delta A_{B_z} \ [\rm{\mu}$rad]")
        # fig.savefig("../fig/sum_mom_delta_A_bz"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # fig, ax = cu.plot_mom(p_min, data_s12['n'], None, cuts, N, label1=label1+" S12"+n_label, s18=True, s18_y=data_s18['n'], s18_y_e=None, weighted=False, pmin=True)
        # # ax.xaxis.set_major_locator(loc)
        # ax.set_ylabel(r"N (entries)")
        # fig.savefig("../fig/sum_mom_N"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # fig, ax = cu.plot_mom(p_min, np.sqrt(data_s12['n']), None, cuts, N, label1=label1+" S12"+n_label, s18=True, s18_y=np.sqrt(data_s18['n']), s18_y_e=None, weighted=False, pmin=True)
        # # ax.xaxis.set_major_locator(loc)
        # ax.set_ylabel(r"$\sqrt{N}$ (entries)")
        # fig.savefig("../fig/sum_mom_sqrt_N"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # # plot A_edm
        # fig, ax = cu.plot_mom(p_min, data_s12['A_edm_blind']*1e3, data_s12['A_edm_blind_e']*1e3, cuts, N, label2=r"$\langle A_{\rm{EDM}}  \rangle =$", label1=label1+" S12", s18=True, s18_y=data_s18['A_edm_blind']*1e3, s18_y_e=data_s18['A_edm_blind_e']*1e3, weighted=False, pmin=True)
        # ax.set_ylabel(r"$A_{\rm{EDM}} \ [\rm{\mu}$rad]")
        # fig.savefig("../fig/sum_mom_A_edm"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # # plot c
        # fig, ax = cu.plot_mom(p_min, data_s12['c']*1e3, data_s12['c_e']*1e3, cuts, N, label2=r"$\langle c \rangle =$", label1=label1+" S12"+n_label, s18=True, s18_y=data_s18['c']*1e3, s18_y_e=data_s18['c_e']*1e3, weighted=False, pmin=True)
        # ax.set_ylabel(r"$c \ [\rm{\mu}$rad]")
        # fig.savefig("../fig/sum_mom_c"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');

        # # plot sigma 
        # fig, ax = cu.plot_mom(p_min, data_s12['sigma_y'], None, cuts, N, weighted=False, label1=label1+" S12"+n_label, s18=True, s18_y=data_s12['sigma_y'], s18_y_e=None, pmin=True)
        # ax.set_ylabel(r"$\sigma_{\theta_y}$  [mrad]")
        # fig.savefig("../fig/sum_mom_sigma"+"_S"+str(station)+".png", dpi=300, bbox_inches='tight');


def plot_all_mom(df=args.file, scan=False):

    data = pd.read_csv(df)

    if(scan==True):

        ds_names=('Run-1a', "Run-1b", "Run-1c", "Run-1d")

        #S1218
        B_z=   data['Bz']
        Bz_e= data['Bz_e']
        N=data_s1218['n']
        ds_colors=["k", "k", "k", "k"]
        ds_markers=["d", "d", "d", "d"]
        A_bz_mean= round(np.sum(A_bz * N)/np.sum(N),1)
        A_bz_mean_e = round(1.0/np.sqrt(np.sum(1.0/A_bz_e**2))  ,1)

        #S12 and S18 
        A_bz_s12=   data_s12['A_Bz']*1e3
        A_bz_e_s12= data_s12['A_Bz_e']*1e3
        ds_colors_s12=["r", "r", "r", "r"]
        ds_markers_s12=["+", "+", "+", "+"]
        A_bz_s18=   data_s18['A_Bz']*1e3
        A_bz_e_s18= data_s18['A_Bz_e']*1e3
        ds_colors_s18=["b", "b", "b", "b"]
        ds_markers_s18=["o", "o", "o", "o"]


        fig, ax = cu.plot_fom(ds_names, A_bz_s12, A_bz_e_s12, ds_colors_s12, ds_markers_s12, y_label=r"$A_{B_z} \ [\rm{\mu}$rad]", eL=" ", label="S12", zorder=1)
        fig, ax = cu.plot_fom(ds_names, A_bz, A_bz_e, ds_colors, ds_markers, y_label=r"$A_{B_z} \ [\rm{\mu}$rad]", fig=fig, ax=ax, eL=" ", label="S1218", zorder=2)
        fig, ax = cu.plot_fom(ds_names, A_bz_s18, A_bz_e_s18, ds_colors_s18, ds_markers_s18, y_label=r"$A_{B_z} \ [\rm{\mu}$rad]", fig=fig, ax=ax, eL=" ", label="S18", zorder=3)

        band_width=2
        ax.set_xlim(0.7, 4.3)
        ax.set_ylim(-35, 45)
        ax.plot([0,5],[A_bz_mean, A_bz_mean], ls=":", c="g", zorder=3, label=r"$\langle A_{B_z} \rangle$="+str(A_bz_mean)+"("+str(A_bz_mean_e)+r") $\rm{\mu}$rad")
        ax.set_xlabel("")
        plt.xticks(fontsize=14)

        ax.add_patch(patches.Rectangle(
                xy=(0, A_bz_mean-A_bz_mean_e),  # point of origin.
                width=5,
                height=A_bz_mean_e*2,
                linewidth=0,
                color='green',
                fill=True,
                alpha=0.7,
                zorder=4,
                label=r"$1\sigma$ band"
            )
        )

        ax.add_patch(patches.Rectangle(
                xy=(0, A_bz_mean-(A_bz_mean_e*band_width)),  # point of origin.
                width=5,
                height=A_bz_mean_e*band_width*2,
                linewidth=0,
                color='blue',
                fill=True,
                alpha=0.2,
                zorder=5,
                label=str(band_width)+r"$\sigma$ band"
            )
        )

        plt.legend(fontsize=11, loc=(0.02,0.62))
        plt.tight_layout()
        fig.savefig("../fig/sum_A_bz_s12s18.png", dpi=200, bbox_inches='tight');



    if(scan==False):

        s12_cut = data['station']==12
        s18_cut = data['station']==12
        s1218_cut = data['station']==1218
           
        data_s12 = data[s12_cut].reset_index()
        data_s18 = data[s18_cut].reset_index()
        data_s1218 = data[s1218_cut].reset_index()

        ds_names=('Run-1a', "Run-1b", "Run-1c", "Run-1d")

        #S1218
        A_bz=   data_s1218['A_Bz']*1e3
        A_bz_e= data_s1218['A_Bz_e']*1e3
        N=data_s1218['n']
        ds_colors=["k", "k", "k", "k"]
        ds_markers=["d", "d", "d", "d"]
        A_bz_mean= round(np.sum(A_bz * N)/np.sum(N),1)
        A_bz_mean_e = round(1.0/np.sqrt(np.sum(1.0/A_bz_e**2))  ,1)

        #S12 and S18 
        A_bz_s12=   data_s12['A_Bz']*1e3
        A_bz_e_s12= data_s12['A_Bz_e']*1e3
        ds_colors_s12=["r", "r", "r", "r"]
        ds_markers_s12=["+", "+", "+", "+"]
        A_bz_s18=   data_s18['A_Bz']*1e3
        A_bz_e_s18= data_s18['A_Bz_e']*1e3
        ds_colors_s18=["b", "b", "b", "b"]
        ds_markers_s18=["o", "o", "o", "o"]


        fig, ax = cu.plot_fom(ds_names, A_bz_s12, A_bz_e_s12, ds_colors_s12, ds_markers_s12, y_label=r"$A_{B_z} \ [\rm{\mu}$rad]", eL=" ", label="S12", zorder=1)
        fig, ax = cu.plot_fom(ds_names, A_bz, A_bz_e, ds_colors, ds_markers, y_label=r"$A_{B_z} \ [\rm{\mu}$rad]", fig=fig, ax=ax, eL=" ", label="S1218", zorder=2)
        fig, ax = cu.plot_fom(ds_names, A_bz_s18, A_bz_e_s18, ds_colors_s18, ds_markers_s18, y_label=r"$A_{B_z} \ [\rm{\mu}$rad]", fig=fig, ax=ax, eL=" ", label="S18", zorder=3)

        band_width=2
        ax.set_xlim(0.7, 4.3)
        ax.set_ylim(-35, 45)
        ax.plot([0,5],[A_bz_mean, A_bz_mean], ls=":", c="g", zorder=3, label=r"$\langle A_{B_z} \rangle$="+str(A_bz_mean)+"("+str(A_bz_mean_e)+r") $\rm{\mu}$rad")
        ax.set_xlabel("")
        plt.xticks(fontsize=14)

        ax.add_patch(patches.Rectangle(
                xy=(0, A_bz_mean-A_bz_mean_e),  # point of origin.
                width=5,
                height=A_bz_mean_e*2,
                linewidth=0,
                color='green',
                fill=True,
                alpha=0.7,
                zorder=4,
                label=r"$1\sigma$ band"
            )
        )

        ax.add_patch(patches.Rectangle(
                xy=(0, A_bz_mean-(A_bz_mean_e*band_width)),  # point of origin.
                width=5,
                height=A_bz_mean_e*band_width*2,
                linewidth=0,
                color='blue',
                fill=True,
                alpha=0.2,
                zorder=5,
                label=str(band_width)+r"$\sigma$ band"
            )
        )

        plt.legend(fontsize=11, loc=(0.02,0.62))
        plt.tight_layout()
        fig.savefig("../fig/sum_A_bz_s12s18.png", dpi=200, bbox_inches='tight');

    

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
            ax=sn.heatmap(df_corr, annot=True, fmt='.3f', linewidths=.5, cmap="bwr")
            cu.textL(ax, 0.5, 1.1, "S"+str(station))
            fig.savefig("../fig/corr_"+key+"_S"+str(station)+".png", dpi=300)


if __name__ == "__main__":
    main()