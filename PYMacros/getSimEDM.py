'''
Author: Gleb Lukicov

Get an estimate of the longitudinal field from simulation or data
using EDM blinding
'''
import numpy as np
import pandas as pd
# MPL in batch mode
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from scipy import optimize
sys.path.append("../CommonUtils/") # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
import RUtils as ru
import argparse 


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--t_min", type=float, default=0) # us 
arg_parser.add_argument("--t_max", type=float, default=300) # us 
arg_parser.add_argument("--p_min", type=float, default=700) # us 
arg_parser.add_argument("--p_max", type=float, default=2400) # us 
arg_parser.add_argument("--bin_w", type=float, default=15) # ns 
arg_parser.add_argument("--g2period", type=float, default=None) # us 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/Sim/Sim.h5") 
args=arg_parser.parse_args()

### Define constants and starting fit parameters
font_size=14 # for plots
stations=(1218)
expected_DSs = ("Sim")

### Get ds_name from filename
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
folder=args.hdf.replace(".","/").split("/")[-3] 
print("Detected DS name:", ds_name, "from the input file!")
if( (folder != "Sim") and (folder != "EDM")): raise Exception("Loaded pre-skimmed simulation or EDM file")
if(not ds_name in expected_DSs): raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")

# Now that we know what DS we have, we can
# set tune and calculate expected FFTs and
cu._DS=ds_name

sim=False
if(ds_name == "Sim"):
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

#binning 
bin_w = args.bin_w*1e-3 # 10 ns 
bin_n = int( g2period/bin_w)
print("Setting bin width of", bin_w*1e3, "ns with ~", bin_n, "bins")

#starting fit parameters and their labels for plotting 
par_names_theta= ["A_edm", "omega", "c"]; par_labels_theta= [r"$A_{\mathrm{EDM}}$", r"$\omega_{a}$", r"$c$"]; par_units_theta=["mrad", "MHz", "mrad"]
if(sim): 
    p0_theta=[ [0.22, 1.31, 0.0]]
print("Starting pars",*par_names_theta, *p0_theta)
global_label=ds_name+"_"
file_label=["_S"+str(s)+"_"+global_label for s in stations]

scan_label=-1

def main():

    data=load_data(args.hdf)

    plot_counts_theta(data)


def load_data(df_path):
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
    print("Total tracks after cuts", round(data_hdf.shape[0]/1e6,2), "M")

    # calculate variables for plotting
    p=data_hdf['trackMomentum']
    py=data_hdf['trackMomentumY']
    t=data_hdf['trackT0']
    theta_y_mrad = np.arctan2(py, p)*1e3 # rad -> mrad
    mod_times = cu.get_g2_mod_time(t, g2period) # Module the g-2 oscillation time 
    data_hdf['mod_times']=mod_times # add to the data frame 
    data_hdf['theta_y_mrad']=theta_y_mrad # add to the data frame 
    
    # select all stations for simulation
    if(sim): data = [data_hdf]

    #split into two stations for data 
    if(not sim): data = [ data_hdf[data_hdf['station'] == 12], data_hdf[data_hdf['station'] == 18] ];
        
    return data


def plot_counts_theta(data):
    
    for i_station, station in enumerate(stations):
        data_station=data[i_station]
        N=data_station.shape[0]
        print("Entries: ", N, " in S"+str(station))


        if(1==1):

            #############
            #Blinded (EDM) fit for B_Z 
            ############      
            ang=data_station['theta_y_mrad']
            tmod_abs=data_station['mod_times']

            ### Digitise data with weights
            xy_bins=(bin_n, bin_n)
            h,xedges,yedges  = np.histogram2d(tmod_abs, ang, bins=xy_bins);
            
            # expand 
            (x_w, y_w), binsXY, dBinXY = ru.hist2np(h, (xedges,yedges))
            print("Got XY bins", binsXY)
            
            #profile
            df_binned =cu.Profile(x_w, y_w, None, nbins=bin_n, xmin=np.min(x_w), xmax=np.max(x_w), mean=True, only_binned=True)
            x, y, y_e, x_e =df_binned['bincenters'], df_binned['ymean'], df_binned['yerr'], df_binned['xerr']

            #Fit
            par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.edm_sim, p0_theta)
            if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")

            #Plot
            if(sim): legend=ds_name+" S"+str(station);
            fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.edm_sim, 
                                         par, par_e, chi2_ndf, ndf, bin_w, N,
                                         t_min, t_max, p_min, p_max,
                                         par_labels_theta, par_units_theta, 
                                         legend_data = legend,
                                         legend_fit=r'Fit: $\langle \theta(t) \rangle =  A_{\mathrm{EDM}}\sin(\omega_a t) + c$',
                                         ylabel=r"$\langle\theta_y\rangle$ [mrad] per "+str(int(bin_w*1e3))+" ns",
                                         font_size=font_size,
                                         prec=2, 
                                         urad=False)
            ax.set_xlim(0, g2period);
            ax.set_ylim(ax.get_ylim()[0]*2.0, ax.get_ylim()[1]*1.6);
            if(not sim): ax.set_ylim(ax.get_ylim()[0]*1.23, ax.get_ylim()[1]*1.4)
            cu.textL(ax, 0.75, 0.15, leg_data, fs=font_size)
            cu.textL(ax, 0.25, 0.17, leg_fit, fs=font_size, c="r")
            print("Fit in "+ds_name+" S:"+str(station), leg_fit)
            plt.tight_layout()
            fig.savefig("../fig/sim_"+ds_name+"_S"+str(station)+".png", dpi=300)
      


if __name__ == '__main__':
    main()