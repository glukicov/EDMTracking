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
import BlindEDM

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--t_min", type=float, default=30.56) # us 
arg_parser.add_argument("--t_max", type=float, default=454.00) # us 
arg_parser.add_argument("--p_min", type=float, default=700) # us 
arg_parser.add_argument("--p_max", type=float, default=2400) # us 
arg_parser.add_argument("--bin_w_count", type=float, default=15) # ns 
arg_parser.add_argument("--bin_w", type=float, default=15) # ns 
arg_parser.add_argument("--g2period", type=float, default=None) # us 
arg_parser.add_argument("--phase", type=float, default=None) # us 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/EDM/60h.h5")
arg_parser.add_argument("--corr", action='store_true', default=False, help="Save covariance matrix for plotting")
arg_parser.add_argument("--scan", action='store_true', default=False, help="if run externally for iterative scans - dump 𝝌2 and fitted pars to a file for summary plots") 
arg_parser.add_argument("--hist", action='store_true', default=False)
args=arg_parser.parse_args()

### Define constants and starting fit parameters
font_size=14 # for plots
stations=(12, 18)
# stations=([1218])
expected_DSs = ("60h", "9D", "HK",   "EG", "Sim", "Bz",    "R1")
official_DSs = ("1a",  "1c",  "1b",  "1d",  "Sim",  "Bz",   "1")

### Get ds_name from filename
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
ds_name_official=official_DSs[expected_DSs.index(ds_name)]
folder=args.hdf.replace(".","/").split("/")[-3] 
print("Detected DS name:", ds_name, ds_name_official, "from the input file!")
if( (folder != "Sim") and (folder != "EDM")): raise Exception("Loaded pre-skimmed simulation or EDM file")
if(not ds_name in expected_DSs): raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")
cu._DS=ds_name
# output scan file, HDF5 keys, file labels  
keys=["count", "theta"]
global_label=ds_name+"_"
file_label=["_S"+str(s)+"_"+global_label for s in stations]
scan_label=-1

#set data or sim key 
sim=False
key_df="QualityVertices"
if(ds_name == "Sim" or ds_name=="Bz"):
    print("Simulation data is loaded!"); 
    sim=True; 
    stations=([1218])
    key_df=None

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
if (sim): t_min=4.3; t_max=200
print("Starting and end times:", t_min, "to", t_max, "us")
p_min = args.p_min # MeV 
p_max = args.p_max # MeV 
print("Momentum cuts:", p_min, "to", p_max, "MeV")
p_min_counts = 1800 # MeV 
p_max_counts = 3100 # MeV 
if(args.scan): p_min_counts=args.p_min
print("Momentum cuts (for counts only):", p_min_counts, "to", p_max_counts, "MeV")

#starting fit parameters and their labels for plotting 
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N_{0}$", r"$\tau$", r"$A$", r"$\phi$"]; par_units_count=[" ",  r"$\rm{\mu}$s", " ", "rad"]
par_names_theta= ["A_Bz", "A_edm_blind", "c"]; par_labels_theta= [r"$A_{B_{z}}$", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$", r"$c$"]; par_units_theta=[r"$\rm{\mu}$rad", r"$\rm{\mu}$rad", r"$\rm{\mu}$rad"]
par_labels_truth=par_labels_theta.copy(); par_labels_truth[1]=r"$A_{\mathrm{EDM}}$"
p0_count=[ [50000, 64, 0.339, 2.07], [50000, 64, 0.339, 2.07]]
p0_theta_blinded=[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
if(sim): 
    p0_count=[ [3000, 64, -0.40, 6.3], [3000, 64, -0.40, 6.3]]
print("Starting pars theta blinded", *par_names_theta, *p0_theta_blinded)
print("Starting pars count",*par_names_count, *p0_count)

### Define global variables (containers)
residuals_counts, residuals_theta, times_counts, times_theta, errors_theta, errors_counts =[ [] ], [ [] ], [ [] ], [ [] ], [ [] ], [ [] ]

def main():

    plot_counts_theta(args.hdf)


def plot_counts_theta(df_path):

    #######
    # Step 1. Get phase 
    #####

    '''
    Load data apply cuts and return two data frames - one per station 
    '''
    print("Opening data...")
    data_hdf = pd.read_hdf(df_path, key=key_df)   #open skimmed 
    print("N before cuts", round(data_hdf.shape[0]/1e6,2), "M") 
    
    #apply cuts 
    mom_cut = ( (data_hdf['trackMomentum'] > p_min_counts) & (data_hdf['trackMomentum'] < p_max_counts) ) # MeV  
    time_cut =( (data_hdf['trackT0'] > t_min) & (data_hdf['trackT0'] < t_max) ) # MeV  
    data_hdf=data_hdf[mom_cut & time_cut]
    data_hdf=data_hdf.reset_index() # reset index from 0 after cuts 
    print("Total tracks after cuts", round(data_hdf.shape[0]/1e6,2), "M")

    #binning 
    bin_w = args.bin_w_count*1e-3 # 10 ns 
    bin_n = int( g2period/bin_w)
    print("Setting bin width of", bin_w*1e3, "ns with ~", bin_n, "bins")

    # calculate variables for plotting
    t=data_hdf['trackT0']
    mod_times = cu.get_g2_mod_time(t, g2period) # Module the g-2 oscillation time 
    data_hdf['mod_times']=mod_times # add to the data frame 

    # select all stations for simulation or when both station are used in the fit
    if(sim or len(stations)==1): data = [data_hdf]

    #split into two stations for data 
    if(not sim and len(stations)==2): data = [ data_hdf[data_hdf['station'] == 12], data_hdf[data_hdf['station'] == 18] ];
    
    #loop over one or two stations
    for i_station, station in enumerate(stations):
        data_station=data[i_station]
        N=data_station.shape[0]
        print("Tracks: ", round(N/1e6,2), "M in S"+str(station))

        #############
        #Plot counts vs. mod time and fit
        #############
        ### Digitise data
        x, y, y_e = cu.get_freq_bin_c_from_data(data_station['mod_times'], bin_w, (0, g2period) )
       
        #Fit
        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.unblinded_wiggle_fixed, p0_count[i_station])
        if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
        if (args.corr): print("Covariance matrix\n", pcov); np.save("../DATA/misc/pcov_count_S"+str(station)+".np", pcov);

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
        if(args.scan==False): fig.savefig("../fig/count_"+ds_name+"_S"+str(station)+".png", dpi=200)

        # if running externally, via a different module and passing scan==True as an argument
        # dump the parameters to a unique file for summary plots
        scan_label="_"+str(t_min)+"_"+str(t_max)+"_"+str(p_min_counts)+"_"+str(p_max_counts)+"_"+str(ndf)
        if(args.scan==True):
            par_dump=np.array([[t_min], t_max, p_min_counts, p_max_counts, chi2_ndf, ndf, g2period, bin_w, N, station, ds_name, *par, *par_e])
            par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "bin_w", "n", "station", "ds"]
            par_dump_keys.extend(par_names_count)
            par_dump_keys.extend( [str(par)+"_e" for par in par_names_count] )
            dict_dump = dict(zip(par_dump_keys,par_dump))
            df = pd.DataFrame.from_records(dict_dump, index='start')
            with open("../DATA/scans/edm_scan_"+keys[0]+".csv", 'a') as f:
                df.to_csv(f, mode='a', header=f.tell()==0)
            #plt.savefig("../fig/scans/count_"+ds_name+"_S"+str(station)+scan_label+".png", dpi=200)
    
        if(args.corr):
        # get residuals for later plots 
            residuals_counts[i_station] = cu.residuals(x, y, cu.unblinded_wiggle_fixed, par)
            times_counts[i_station] = x
            errors_counts[i_station] = y_e

        ### Set constant phase from the fit for the next step
        if(args.phase == None):
            cu._phi=par[-1]
       
        # if phase if passed just set it for the next step
        # phase is passed for scans of entire R-1 dataset in both stations
        else:
            if(len(stations)!=1): raise Exception("Constant single phase is passed for S1218, but looping over both!")
            cu._phi=args.phase
        print("Phase set to", round(cu._phi,5), "rad")

    ############
    # Step 2: Theta fits
    ###########   
    '''
    Load data and apply new cuts and return two data frames - one per station 
    '''
    print("Opening data...")
    data_hdf = pd.read_hdf(df_path, key=key_df)   #open skimmed 
    print("N before cuts", round(data_hdf.shape[0]/1e6,2), "M")
    
    #apply cuts 
    mom_cut = ( (data_hdf['trackMomentum'] > p_min) & (data_hdf['trackMomentum'] < p_max) ) # MeV  
    time_cut =( (data_hdf['trackT0'] > t_min) & (data_hdf['trackT0'] < t_max) ) # MeV  
    data_hdf=data_hdf[mom_cut & time_cut]
    data_hdf=data_hdf.reset_index() # reset index from 0 after cuts 
    N=data_hdf.shape[0]
    print("Total tracks after cuts", round(N/1e6,2), "M")

    #binning 
    bin_w = args.bin_w*1e-3 # 10 ns 
    bin_n = int( g2period/bin_w)
    print("Setting bin width of", bin_w*1e3, "ns with ~", bin_n, "bins")

    # calculate variables for plotting
    p=data_hdf['trackMomentum']
    py=data_hdf['trackMomentumY']
    theta_y_mrad = np.arctan2(py, p)*1e3 # rad -> mrad
    data_hdf['theta_y_mrad']=theta_y_mrad # add to the data frame 
    t=data_hdf['trackT0']
    mod_times = cu.get_g2_mod_time(t, g2period) # Module the g-2 oscillation time 
    data_hdf['mod_times']=mod_times # add to the data frame 

    # select all stations for simulation
    if(sim or len(stations)==1): data = [data_hdf]

    #split into two stations for data 
    if(not sim and len(stations)==2): data = [ data_hdf[data_hdf['station'] == 12], data_hdf[data_hdf['station'] == 18] ];
     
    for i_station, station in enumerate(stations):
        data_station=data[i_station]
        N=data_station.shape[0]
        print("Tracks: ", round(N/1e6,2), "M in S"+str(station))

        #############
        #Blinded (EDM) fit for B_Z 
        ############      
        ### Resolve angle and times     

        # blinding the angle 
        ang_Blinded = data_station['theta_y_mrad'] + BlindEDM.get_delta_blind(blindString=ds_name_official+str(station)) * np.sin( cu._omega * data_station['mod_times'] + cu._phi)
        
        #profile
        df_binned =cu.Profile(data_station['mod_times'], ang_Blinded, None, nbins=bin_n, xmin=np.min(data_station['mod_times']), xmax=np.max(data_station['mod_times']), mean=True, only_binned=True)
        x, y, y_e, x_e =df_binned['bincenters'], df_binned['ymean'], df_binned['yerr'], df_binned['xerr']

        #Fit
        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.thetaY_phase, p0_theta_blinded[i_station])
        if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
        if(args.corr): print("Covariance matrix\n", pcov); np.save("../DATA/misc/pcov_theta_S"+str(station)+".np", pcov);

        #Set legend title for the plot 
        if(sim): legend=ds_name_official+" S"+str(station);
        else:    legend="Run-"+ds_name_official+" dataset S"+str(station);     
        fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.thetaY_phase, 
                                     par, par_e, chi2_ndf, ndf, bin_w, N,
                                     t_min, t_max, p_min, p_max,
                                     par_labels_theta, par_units_theta, 
                                     legend_data = legend,
                                     legend_fit=r'Fit: $\langle \theta(t) \rangle =  A_{\mathrm{B_z}}\cos(\omega_a t + \phi) + A_{\mathrm{EDM}}\sin(\omega_a t + \phi) + c$',
                                     ylabel=r"$\langle\theta_y\rangle$ [mrad] per "+str(int(bin_w*1e3))+" ns",
                                     font_size=font_size,
                                     prec=2)
        ax.set_xlim(0, g2period);
        if(ds_name=="9D"): 
            ax.set_ylim(-0.95, 0.20)
        elif(ds_name=="R1"):
            ax.set_ylim(-0.66, -0.16)
        elif(ds_name=="EG"): 
            ax.set_ylim(-0.9, -0.2)
        elif(ds_name=="HK"): 
            ax.set_ylim(-0.90, 0.35)
        else:
            ax.set_ylim(-0.90, 0.35);
        if(sim): ax.set_ylim(-2.9, 2.5)
        cu.textL(ax, 0.75, 0.15, leg_data, fs=font_size)
        cu.textL(ax, 0.27, 0.17, leg_fit, fs=font_size, c="r")
        print("Fit in "+ds_name+" S:"+str(station), leg_fit)
        if(args.scan==False): fig.savefig("../fig/bz_"+ds_name+"_S"+str(station)+".png", dpi=200)

        if(args.scan==True):
            par_dump=np.array([[t_min], t_max, p_min, p_max, chi2_ndf, ndf, g2period, cu._LT, cu._phi,  bin_w, bin_n, xy_bins[0], xy_bins[1], N, station, ds_name, *par, *par_e])
            par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "lt", "phase", "bin_w", "bin_n", "ndf_x", "ndf_y", "n", "station", "ds"]
            par_dump_keys.extend(par_names_theta)
            par_dump_keys.extend( [str(par)+"_e" for par in par_names_theta] )
            dict_dump = dict(zip(par_dump_keys,par_dump))
            df = pd.DataFrame.from_records(dict_dump, index='start')
            with open("../DATA/scans/edm_scan_"+keys[1]+".csv", 'a') as f:
                df.to_csv(f, mode='a', header=f.tell()==0)
            #plt.savefig("../fig/scans/bz_"+ds_name+"_S"+str(station)+scan_label+".png", dpi=200)

        # get residuals for later plots 
        if(args.corr):
            residuals_theta[i_station] = cu.residuals(x, y, cu.thetaY_phase, par)
            times_theta[i_station] = x
            errors_theta[i_station] = y_e

        #############
        # Make truth (un-blinded fits) if simulation
        #############
        if(sim or 1==1):
            print("Making truth plots in simulation")

            # Bin 
            df_binned =cu.Profile(data_station['mod_times'], data_station['theta_y_mrad'], None, nbins=bin_n, xmin=np.min(data_station['mod_times']), xmax=np.max(data_station['mod_times']), mean=True, only_binned=True)
            x, y, y_e, x_e =df_binned['bincenters'], df_binned['ymean'], df_binned['yerr'], df_binned['xerr']

            # Fit 
            par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.thetaY_phase, p0_theta_blinded[i_station])
            if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
            if(args.corr): print("Covariance matrix", pcov); np.save("../DATA/misc/pcov_truth_S"+str(station)+".np", pcov);

            #Plot
            fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.thetaY_phase, 
                                     par, par_e, chi2_ndf, ndf, bin_w, N,
                                     t_min, t_max, p_min, p_max,
                                     par_labels_truth, par_units_theta, 
                                     legend_data = legend,
                                     legend_fit=r'Fit: $\langle \theta(t) \rangle =  A_{\mathrm{B_z}}\cos(\omega_a t + \phi) + A_{\mathrm{EDM}}\sin(\omega_a t + \phi) + c$',
                                     ylabel=r"$\langle\theta_y\rangle$ [mrad] per "+str(int(bin_w*1e3))+" ns",
                                     font_size=font_size,
                                     prec=2)
            cu.textL(ax, 0.75, 0.15, leg_data, fs=font_size)
            cu.textL(ax, 0.27, 0.17, leg_fit, fs=font_size, c="r")
            ax.set_xlim(0, g2period);
            ax.set_ylim(-0.80, 0.55);
            if(sim): ax.set_ylim(-2.9, 2.5);
            if(args.scan==False): fig.savefig("../fig/bz_truth_"+ds_name+"_S"+str(station)+".png", dpi=200)

    ###
    #End of stations loop
    ##

    #make sanity plots 
    if(args.hist):

        fig, _ = plt.subplots()
        bin_w_mom = 10 
        mom=data_station['trackMomentum']
        n_bins_mom=int(round((max(mom)-min(mom))/bin_w_mom,2))
        ax, _ = cu.plotHist(mom, n_bins=n_bins_mom, prec=3, units="MeV", label="Run-"+ds_name_official+" dataset S"+str(station) )
        legend=cu.legend5(*cu.stats5(mom), "MeV", prec=2)
        cu.textL(ax, 0.76, 0.85, str(legend), fs=14)
        ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.1)
        # ax.set_xlim(-50,50)
        ax.set_xlabel(r"$p$ [MeV]", fontsize=font_size);
        ax.set_ylabel("Entries per "+str(bin_w_mom)+" MeV", fontsize=font_size);
        ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.26, 1.0))
        fig.savefig("../fig/mom_"+ds_name+"_S"+str(station)+".png", dpi=200, bbox_inches='tight')
        
        fig, _ = plt.subplots()
        n_bins_ang=400*2
        ax, _ = cu.plotHist(ang, n_bins=n_bins_ang, prec=3, units="mrad", label="Run-"+ds_name_official+" dataset S"+str(station) )
        legend=cu.legend3_sd(*cu.stats3_sd(ang), "mrad", prec=3)
        cu.textL(ax, 0.8, 0.85, str(legend), fs=14)
        ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.1)
        ax.set_xlim(-50,50)
        ax.set_xlabel(r"$\theta_y$ [mrad]", fontsize=font_size);
        ax.set_ylabel("Entries per "+str(round((max(ang)-min(ang))/n_bins_ang,3))+" mrad", fontsize=font_size);
        ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.3, 1.0))
        fig.savefig("../fig/theta_"+ds_name+"_S"+str(station)+".png", dpi=200, bbox_inches='tight')
        
    #-------end of looping over stations

    ## now if not scanning - get FFTs for both stations
    ## FFTs
    if(args.corr):
        print("Plotting residuals and FFTs...")
        cu.residual_plots(times_counts, residuals_counts, sim=sim, eL="count", file_label=file_label)
        cu.residual_plots(times_theta, residuals_theta, sim=sim, eL="theta",  file_label=file_label)
        cu.pull_plots(residuals_theta, errors_theta, file_label=file_label  , eL="theta")
        cu.pull_plots(residuals_counts, errors_counts, file_label=file_label, eL="count")
if __name__ == '__main__':
    main()