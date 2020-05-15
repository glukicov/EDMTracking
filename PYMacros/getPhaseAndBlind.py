'''
Author: Gleb Lukicov (g.lukicov@ucl.ac.uk)
Updated: 12 May 2020
Purpose: Get an estimate of the longitudinal field from simulation or data with EDM blinding
'''
import os, sys
import argparse 
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') # MPL in batch mode
import matplotlib.pyplot as plt
from scipy import optimize
sys.path.append("../CommonUtils/") # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu 
# Import blinding libraries 
import BlindEDM # https://github.com/glukicov/EDMTracking/blob/master/PYMacros/BlindEDM.py

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--t_min", type=float, default=30.56, help="Fit start-time [us]") 
arg_parser.add_argument("--t_max", type=float, default=454.00, help="Fit end-time [us]") 
arg_parser.add_argument("--p_min_count", type=float, default=1800, help="Min momentum cut [MeV]")
arg_parser.add_argument("--p_max_count", type=float, default=3100, help="Max momentum cut [MeV]")
arg_parser.add_argument("--bin_w_count", type=float, default=15, help="Bin width for counts plot [ns]")
arg_parser.add_argument("--g2period", type=float, default=None, help="g-2 period, if none BNL value is used [us]") 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/EDM/60h.h5", help="Full path to the data file: HDF5")
arg_parser.add_argument("--both", action='store_true', default=False, help="Separate fists for S12 and S18")
arg_parser.add_argument("--phase", action='store_true', default=False, help="Determine and plot phase")
arg_parser.add_argument("--blind", action='store_true', default=False, help="Blind data")

args=arg_parser.parse_args()

### Define constants and starting fit parameters
font_size=14 # for plots
#default is both stations in the fit, if both passed loop over S12 and S18 separately 
stations=([1218])
if (args.both): stations=(12, 18)
# Only allow expected input data - convert to official naming standard 
expected_DSs = ("60h", "9D", "HK",   "EG", "Sim", "Bz",    "R1")
official_DSs = ("1a",  "1c",  "1b",  "1d",  "Sim",  "Bz",   "1")

### Get ds_name from filename + string magic 
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
ds_name_official=official_DSs[expected_DSs.index(ds_name)]
folder=args.hdf.replace(".","/").split("/")[-3] 
print("Detected DS name:", ds_name, ds_name_official, "from the input file!")
if( (folder != "Sim") and (folder != "EDM")): raise Exception("Load a pre-skimmed simulation or an EDM file")
if(not ds_name in expected_DSs): raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")
cu._DS=ds_name
# output scan file, HDF5 keys, file labels, scan labels 
keys=["count", "theta"] # for scans 
global_label=ds_name+"_" # for plots 
file_label=["_S"+str(s)+"_"+global_label for s in stations] 
scan_label=None # for scans 

#set data or sim key (key here refers to the HDF5 key in the data file e.g. "Tree name")
sim=False
key_df="QualityVertices"
if(ds_name == "Sim" or ds_name=="Bz"):
    print("Simulation data is loaded!"); 
    sim=True; 
    stations=([1218])
    key_df=None

#Set gm2 period and omega from BNL, unless scanning  
if(args.g2period == None): 
    g2period = round(1/cu._f_a,6)  # 4.365411 us (arXiv:hep-ex/0602035) 
else: 
    g2period=args.g2period;  
print("g-2 period ", g2period, "us")
cu._omega=round(2*np.pi/g2period, 6) # rad/us (magic) 
print("Magic omega set to", cu._omega, "MHz")

#Resolve cuts from arguments 
t_min = args.t_min # us 
t_max = args.t_max # us 
if (sim): t_min=4.3; t_max=200
print("Starting and end times:", t_min, "to", t_max, "us")
p_min_counts = args.p_min_count # MeV 
p_max_counts = args.p_max_count # MeV 
# if(args.scan): p_min_counts=args.p_min
print("Momentum cuts (for counts only):", p_min_counts, "to", p_max_counts, "MeV")

#starting fit parameters and their labels (+LaTeX) for plotting 
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N_{0}$", r"$\tau$", r"$A$", r"$\phi$"]; par_units_count=[" ",  r"$\rm{\mu}$s", " ", "rad"]
p0_count=[ [50000, 64, 0.339, 2.07], [50000, 64, 0.339, 2.07]]
if(sim): 
    p0_count=[ [3000, 64, -0.40, 6.3], [3000, 64, -0.40, 6.3]]
print("Starting pars count",*par_names_count, *p0_count)


def main():

    if(args.phase): plot_phase(args.hdf)
    if(args.blind): blind_data(args.hdf)


def blind_data(df_path):
    cu._phi = cu.get_phase(ds_name)

    print("Opening data...")
    data_hdf = pd.read_hdf(df_path, key=key_df)   #open skimmed 
    print("Total tracks before cuts", round(data_hdf.shape[0]/1e6,2), "M")

    # calculate variables for plotting
    p=data_hdf['trackMomentum']
    py=data_hdf['trackMomentumY']
    theta_y_mrad = np.arctan2(py, p)*1e3 # rad -> mrad
    mod_time = cu.get_g2_mod_time(data_hdf['trackT0'], g2period) # Module the g-2 oscillation time     
    # blinding the angle 
    ang_Blinded = theta_y_mrad + BlindEDM.get_delta_blind(blindString=ds_name_official) * np.sin( cu._omega * mod_time + cu._phi)
    data_hdf['ang_Blinded']=ang_Blinded
    data_hdf['mod_time']=theta_y_mrad
    data_hdf=data_hdf.drop(['trackMomentumY'], axis=1)
    
    data_hdf.to_hdf("../DATA/HDF/EDM/Blinded/"+ds_name+".h5", key=key_df, mode='a', complevel=9, complib="zlib", format="fixed")

def plot_phase(df_path):

    ### load data once, and apply two independent set of cuts to the copies of data
    print("Opening data...")
    data_hdf = pd.read_hdf(df_path, key=key_df)   #open skimmed 
    print("Total tracks before cuts", round(data_hdf.shape[0]/1e6,2), "M") 

    # select all stations for simulation or when both station are used in the fit
    if(sim or len(stations)==1): data = [data_hdf]

    #split into two stations for data 
    if(not sim and len(stations)==2): data = [ data_hdf[data_hdf['station'] == 12], data_hdf[data_hdf['station'] == 18] ];

    #loop over one or two stations
    for i_station, station in enumerate(stations):  

        data_station=data[i_station]
        print("Tracks before cuts: ", round(data_station.shape[0]/1e6,2), "M in S"+str(station))    

        #######
        # Step 1. Get phase 
        #####

        #  if a constant phase is not passed - determine one from data

        #apply cuts 
        mom_cut = ( (data_station['trackMomentum'] > p_min_counts) & (data_station['trackMomentum'] < p_max_counts) ) # MeV  
        time_cut =( (data_station['trackT0'] > t_min) & (data_station['trackT0'] < t_max) ) # MeV  
        data_station_count=data_station[mom_cut & time_cut]
        data_station_count=data_station_count.reset_index() # reset index from 0 after cuts 
        N = data_station_count.shape[0]
        print("Total tracks after count cuts", round(N/1e6,2), "M in S"+str(station))
        
        #binning 
        bin_w = args.bin_w_count*1e-3 # 10 ns 
        bin_n = int( g2period/bin_w)
        print("Setting bin width of", bin_w*1e3, "ns with ~", bin_n, "bins")

        # calculate variables for plotting
        mod_time = cu.get_g2_mod_time(data_station_count['trackT0'], g2period) # Module the g-2 oscillation time 

        ### Digitise data
        x, y, y_e = cu.get_freq_bin_c_from_data(mod_time, bin_w, (0, g2period) )
       
        #Fit
        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.unblinded_wiggle_fixed, p0_count[i_station])
        if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
    
        #Set legend title for the plot 
        if(sim): legend=ds_name_official+" S"+str(station);
        else:    legend="Run-"+ds_name_official+" dataset S"+str(station);  
        if(ds_name=="R1" or  ds_name=="EG"): ms_ds=2                   
        else: ms_ds=2                   
        fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.unblinded_wiggle_fixed, 
                                     par, par_e, chi2_ndf, ndf, bin_w, N,
                                     t_min, t_max, p_min_counts, p_max_counts,
                                     par_labels_count, par_units_count, 
                                     legend_data = legend,
                                     legend_fit=r'Fit: $N(t)=N_{0}e^{-t/\tau}[1+A\cos(\omega_at+\phi)]$',
                                     ylabel=r"Counts ($N$) per "+str(int(bin_w*1e3))+" ns",
                                     font_size=font_size,
                                     lw=2,
                                     marker=".",
                                     ms=ms_ds,
                                     prec=3,
                                     urad=False)
        
        ax.set_ylim(np.amin(y)*0.9, np.amax(y)*1.15);
        if(sim): ax.set_ylim(np.amin(y)*0.9, np.amax(y)*1.4); cu.textL(ax, 0.5, 0.2, leg_fit, c="r", fs=font_size+1); cu.textL(ax, 0.80, 0.70, leg_data, fs=font_size+1)
        if(not sim): cu.textL(ax, 0.65, 0.20, leg_fit, c="r", fs=font_size+1); cu.textL(ax, 0.23, 0.65, leg_data, fs=font_size+1)
        ax.set_xlim(0, g2period);
        fig.savefig("../fig/count_"+ds_name+"_S"+str(station)+".png", dpi=200)

        ### Set constant phase from the fit for the next step
        #cu._phi=par[-1]
        print("Phase is", round(par[-1], 4), "in dataset", ds_name, ds_name_official, 'in', station)
               

if __name__ == '__main__':
    main()