'''
Author: Gleb Lukicov

Get an estimate of the longitudinal field from simulation or data
using EDM blinding
'''
import numpy as np
import pandas as pd
import os, sys
from scipy import optimize
sys.path.append("../CommonUtils/") # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
import RUtils as ru
import argparse 


arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--t_min", type=float, default=4.3) # us 
arg_parser.add_argument("--t_min", type=float, default=30.56) # us 
arg_parser.add_argument("--t_max", type=float, default=454.00) # us 
arg_parser.add_argument("--p_min", type=float, default=1800) # us 
arg_parser.add_argument("--p_max", type=float, default=3100) # us 
arg_parser.add_argument("--bin_w", type=float, default=15) # ns 
# arg_parser.add_argument("--bin_w", type=float, default=150.5314) # ns 
# arg_parser.add_argument("--bin_w", type=float, default=149.2) # ns 
arg_parser.add_argument("--g2period", type=float, default=None) # us 
arg_parser.add_argument("--phase", type=float, default=None) # us 
arg_parser.add_argument("--lt", type=float, default=None) # us 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/EDM/R1.h5") 
# arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/Sim/Bz.h5") 
# arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/Sim/Sim.h5") 
# arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/EDM/60h.h5", help="input data")
arg_parser.add_argument("--scan", action='store_true', default=False, help="if run externally for iterative scans - dump 𝝌2 and fitted pars to a file for summary plots") 
arg_parser.add_argument("--count", action='store_true', default=False)
args=arg_parser.parse_args()

### Define constants and starting fit parameters
font_size=14 # for plots
# stations=(12, 18)
stations=([1218])
expected_DSs = ("60h", "9D", "HK",   "EG", "Sim", "Bz",    "R1")
official_DSs = ("1a",  "1c",  "1b",  "1d",  "Sim",  "Bz",   "1")

### Get ds_name from filename
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
ds_name_official=official_DSs[expected_DSs.index(ds_name)]
folder=args.hdf.replace(".","/").split("/")[-3] 
print("Detected DS name:", ds_name, ds_name_official, "from the input file!")
if( (folder != "Sim") and (folder != "EDM")): raise Exception("Loaded pre-skimmed simulation or EDM file")
if(not ds_name in expected_DSs): raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")

# Now that we know what DS we have, we can
# set tune and calculate expected FFTs and
cu._DS=ds_name
print("Setting tune parameters for ", ds_name, "DS")

sim=False
urad_bool=True
if(ds_name == "Sim" or ds_name=="Bz"):
    print("Simulation data is loaded!"); sim=True; stations=([1218])

#Set gm2 period 
if(args.g2period == None): 
    g2period = round(1/0.2290735,6)  # 4.365411 us 
else: 
    g2period=args.g2period;  
print("g-2 period ", g2period, "us")
cu._omega=round(2*np.pi/g2period, 6) # rad/us (magic) 
print("Magic omega set to", cu._omega, "MHz")

#Cuts 
t_min = args.t_min # us 
t_max = args.t_max # us 
print("Starting and end times:", t_min, "to", t_max, "us")
p_min = args.p_min # MeV 
p_max = args.p_max # MeV 
print("Momentum cuts:", p_min, "to", p_max, "MeV")
p_min_counts = 1800 # MeV 
p_max_counts = 3100 # MeV 
print("Momentum cuts (for counts only):", p_min_counts, "to", p_max_counts, "MeV")

#binning 
bin_w = args.bin_w*1e-3 # 10 ns 
bin_n = int( g2period/bin_w)
print("Setting bin width of", bin_w*1e3, "ns with ~", bin_n, "bins")

#starting fit parameters and their labels for plotting 
par_names_theta= ["A_Bz", "A_edm_blind", "c"]; par_labels_theta= [r"$A_{B_{z}}$", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$", r"$c$"]; par_units_theta=[r"$\rm{\mu}$rad", r"$\rm{\mu}$rad", r"$\rm{\mu}$rad"]
p0_theta_blinded=[ [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
print("Starting pars theta blinded", *par_names_theta, *p0_theta_blinded)

### Define global variables
residuals_counts, residuals_theta, times_counts, times_theta, errors_theta, errors_counts =[ [] ], [ [] ], [ [] ], [ [] ], [ [] ], [ [] ]

# output scan file HDF5 keys 
keys=["count", "theta", "truth"]
global_label=ds_name+"_"
file_label=["_S"+str(s)+"_"+global_label for s in stations]

scan_label=-1

def main():

    ### Set constant phase for the next step
    print("\n Starting fast scans for theta only\n")
    if(args.lt == None): 
        cu._LT=61.04
    else:
        cu._LT=args.lt
    print("LT set to", round(cu._LT,2), "us")
    if(args.phase == None):
        cu._phi=2.06974
    else:
        cu._phi=args.phase
    print("Phase set to", round(cu._phi,5), "rad")

    plot_theta(args.hdf)


def plot_theta(df_path):

    '''
    Load data apply cuts and return two data frames - one per station 
    '''
    print("Opening data...")
    data_hdf = pd.read_hdf(df_path)   #open skimmed 
    print("N before cuts", data_hdf.shape[0])
    
    #apply cuts 
    mom_cut = ( (data_hdf['trackMomentum'] > p_min) & (data_hdf['trackMomentum'] < p_max) ) # MeV  
    time_cut =( (data_hdf['trackT0'] > t_min) & (data_hdf['trackT0'] < t_max) ) # MeV  
    data_hdf=data_hdf[mom_cut & time_cut]
    data_hdf=data_hdf.reset_index() # reset index from 0 after cuts 
    N=data_hdf.shape[0]
    print("Total tracks after cuts", round(N/1e6,2), "M")

    # calculate variables for plotting
    p=data_hdf['trackMomentum']
    py=data_hdf['trackMomentumY']
    theta_y_mrad = np.arctan2(py, p)*1e3 # rad -> mrad
    data_hdf['theta_y_mrad']=theta_y_mrad # add to the data frame 

    # select all stations for simulation
    if(sim or len(stations)==1): data = [data_hdf]

    #split into two stations for data 
    if(not sim and len(stations)==2): data = [ data_hdf[data_hdf['station'] == 12], data_hdf[data_hdf['station'] == 18] ];
     
    for i_station, station in enumerate(stations):
        data_station=data[i_station]
        N=data_station.shape[0]
        print("Entries: ", N, " in S"+str(station))

        #############
        #Blinded (EDM) fit for B_Z 
        ############      
        ### Resolve angle and times
        tmod_abs, weights=cu.get_abs_times_weights(data_station['trackT0'], g2period)
        ang=data_station['theta_y_mrad']

        ### Digitise data with weights
        xy_bins=(bin_n, bin_n)
        h,xedges,yedges  = np.histogram2d(tmod_abs, ang, weights=weights, bins=xy_bins);
        
        # expand 
        (x_w, y_w), binsXY, dBinXY = ru.hist2np(h, (xedges,yedges))
        print("Got XY bins", binsXY)
        
        #profile
        df_binned =cu.Profile(x_w, y_w, None, nbins=bin_n, xmin=np.min(x_w), xmax=np.max(x_w), mean=True, only_binned=True)
        x, y, y_e, x_e =df_binned['bincenters'], df_binned['ymean'], df_binned['yerr'], df_binned['xerr']

        #Fit
        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.thetaY_phase, p0_theta_blinded[i_station])
        if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
        

        if(args.scan==True):
            par_dump=np.array([[t_min], t_max, p_min, p_max, chi2_ndf, ndf, g2period, cu._LT, cu._phi,  bin_w, bin_n, xy_bins[0], xy_bins[1], N, station, ds_name, *par, *par_e])
            par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "lt", "phase", "bin_w", "bin_n", "ndf_x", "ndf_y", "n", "station", "ds"]
            par_dump_keys.extend(par_names_theta)
            par_dump_keys.extend( [str(par)+"_e" for par in par_names_theta] )
            dict_dump = dict(zip(par_dump_keys,par_dump))
            df = pd.DataFrame.from_records(dict_dump, index='start')
            with open("../DATA/scans/edm_scan_"+keys[1]+".csv", 'a') as f:
                df.to_csv(f, mode='a', header=f.tell()==0)
            #plt.savefig("../fig/scans/bz_"+ds_name+"_S"+str(station)+scan_label+".png", dpi=300)

if __name__ == '__main__':
    main()