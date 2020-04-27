'''
Author: Gleb Lukicov
Get an estimate of the longitudinal field from simulation
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') # MPL in batch mode
import matplotlib.pyplot as plt
import os, sys
from scipy import optimize
sys.path.append("../CommonUtils/") # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
import RUtils as ru
import argparse 

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--t_min", type=float, default=4.3) # us 
arg_parser.add_argument("--t_max", type=float, default=100.00) # us 
arg_parser.add_argument("--p_min", type=float, default=1800) # us 
arg_parser.add_argument("--p_max", type=float, default=3100) # us 
arg_parser.add_argument("--bin_w", type=float, default=20) # ns 
arg_parser.add_argument("--g2period", type=float, default=None) # us 
arg_parser.add_argument("--phase", type=float, default=None) # us 
arg_parser.add_argument("--lt", type=float, default=None) # us 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/Sim/Bz.h5") 
arg_parser.add_argument("--scan", action='store_true', default=False, help="if run externally for iterative scans - dump 𝝌2 and fitted pars to a file for summary plots") 
args=arg_parser.parse_args()

### Define constants and starting fit parameters
font_size=14 # for plots
station=1218
expected_DSs = ("Bz")

### Get ds_name from filename
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
folder=args.hdf.replace(".","/").split("/")[-3] 
print("Detected DS name:", ds_name, "from the input file!")
if( (folder != "Sim") and (folder != "EDM")): raise Exception("Loaded pre-skimmed simulation or EDM file")
if(not ds_name in expected_DSs): raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")

# Now that we know what DS we have, we can
# set tune and calculate expected FFTs and
cu._DS=ds_name
print("Setting tune parameters for ", ds_name, "DS")
ds_name+="_Sim"

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
print("Momentum cuts (for theta only):", p_min, "to", p_max, "MeV")
p_min_counts = 1800 # MeV 
p_max_counts = 3100 # MeV 
print("Momentum cuts (for counts only):", p_min_counts, "to", p_max_counts, "MeV")

#binning 
bin_w = args.bin_w*1e-3 # 10 ns 
bin_n = int( g2period/bin_w)
print("Setting bin width of", bin_w*1e3, "ns with ~", bin_n, "bins")

#starting fit parameters and their labels for plotting 
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N$", r"$\tau$", r"$A$", r"$\phi$"]; par_units_count=[" ",  r"$\rm{\mu}$s", " ", "rad"]
par_names_theta= ["A_Bz", "A_edm", "c"]; par_labels_theta= [r"$A_{B_{z}}$", r"$A_{\mathrm{EDM}}$", r"$c$"]; par_units_theta=["mrad", "mrad", "mrad"]
p0_count= [3000, 64.4, -0.40, 6.240]
p0_theta= [0.17, 0.0, 0.0]
urad_bool=False
print("Starting pars count",*par_names_count, *p0_count)
print("Starting pars theta blinded", *par_names_theta, *p0_theta)

# output scan file HDF5 keys 
keys=["count", "theta"]
scan_label=-1

def main():

    #load data for count plot and plot
    plot_counts(args.hdf)
    
    #load data again - use different cuts this time
    plot_theta(args.hdf)



def plot_counts(df_path):
    
    '''
    Load data apply cuts and return two data frames - one per station 
    '''
    print("Opening data...")
    data_hdf = pd.read_hdf(df_path, "trackerNTup/tracker")   #open skimmed 
    print("N before cuts", data_hdf.shape[0])
    
    #apply cuts 
    mom_cut = ( (data_hdf['trackMomentum'] > p_min_counts) & (data_hdf['trackMomentum'] < p_max_counts) ) # MeV  
    time_cut =( (data_hdf['trackT0'] > t_min) & (data_hdf['trackT0'] < t_max) ) # MeV  
    data_hdf=data_hdf[mom_cut & time_cut]
    data_hdf=data_hdf.reset_index() # reset index from 0 after cuts 
    N=data_hdf.shape[0]
    print("Total tracks after cuts", round(N/1e6,2), "M")

    # calculate variables for plotting
    t=data_hdf['trackT0']
    mod_times = cu.get_g2_mod_time(t, g2period) # Module the g-2 oscillation time 
    data_hdf['mod_times']=mod_times # add to the data frame 

    #############
    #Plot counts vs. mod time and fit
    #############
    ### Digitise data
    x, y, y_e = cu.get_freq_bin_c_from_data(mod_times, bin_w, (0, g2period) )
   
    #Fit
    par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.unblinded_wiggle_fixed, p0_count)
    if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")

    #Plot
    legend=ds_name+" S"+str(station);             
    fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.unblinded_wiggle_fixed, 
                                 par, par_e, chi2_ndf, ndf, bin_w, N,
                                 t_min, t_max, p_min_counts, p_max_counts,
                                 par_labels_count, par_units_count, 
                                 legend_data = legend,
                                 legend_fit=r'Fit: $N(t)=N_{0}e^{-t/\tau}[1+A\cos(\omega_at+\phi)]$',
                                 ylabel=r"Counts ($N$) per "+str(int(bin_w*1e3))+" ns",
                                 font_size=font_size,
                                 prec=3)
    
    ax.set_ylim(np.amin(y)*0.9, np.amax(y)*1.4); cu.textL(ax, 0.5, 0.2, leg_fit, c="r", fs=font_size+1); cu.textL(ax, 0.80, 0.70, leg_data, fs=font_size+1)
   
    ax.set_xlim(0, g2period);
    if(args.scan==False): fig.savefig("../fig/count_"+ds_name+"_S"+str(station)+".png", dpi=300)

    # if running externally, via a different module and passing scan==True as an argument
    # dump the parameters to a unique file for summary plots
    scan_label="_"+str(t_min)+"_"+str(t_max)+"_"+str(p_min)+"_"+str(p_max)+"_"+str(ndf)
    if(args.scan==True):
        par_dump=np.array([[t_min], t_max, p_min, p_max, chi2_ndf, ndf, g2period, bin_w, N, station, ds_name, *par, *par_e])
        par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "bin_w", "n", "station", "ds"]
        par_dump_keys.extend(par_names_count)
        par_dump_keys.extend( [str(par)+"_e" for par in par_names_count] )
        dict_dump = dict(zip(par_dump_keys,par_dump))
        df = pd.DataFrame.from_records(dict_dump, index='start')
        with open("../DATA/scans/edm_scan_"+keys[0]+".csv", 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0)
        plt.savefig("../fig/scans/count_"+ds_name+"_S"+str(station)+scan_label+".png", dpi=300)

    ### Set constant phase for the next step
    if(args.lt == None): 
        cu._LT=par[1]
    else:
        cu._LT=args.lt
    print("LT set to", round(cu._LT,2), "us")
    if(args.phase == None):
        cu._phi=par[-1]
    else:
        cu._phi=args.phase
    print("Phase set to", round(cu._phi,2), "rad")    


def plot_theta(df_path):
    
    '''
    Load data apply cuts and return two data frames - one per station 
    '''
    print("Opening data...")
    data_hdf = pd.read_hdf(df_path, "trackerNTup/tracker")   #open skimmed 
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
    t=data_hdf['trackT0']
    theta_y_mrad = np.arctan2(py, p)*1e3 # rad -> mrad
    mod_times = cu.get_g2_mod_time(t, g2period) # Module the g-2 oscillation time 

    #############
    # Make truth (un-blinded fits) if simulation
    #############
    print("Making truth plots in simulation")

    # Bin 
    df_binned =cu.Profile(mod_times,  theta_y_mrad, None, nbins=bin_n, xmin=np.min(mod_times), xmax=np.max(mod_times), mean=True, only_binned=True)
    x, y, y_e, x_e =df_binned['bincenters'], df_binned['ymean'], df_binned['yerr'], df_binned['xerr']

    # Fit 
    par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.thetaY_phase, p0_theta)
    if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")

    #Plot
    legend=ds_name+" S"+str(station);
    fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.thetaY_phase, 
                             par, par_e, chi2_ndf, ndf, bin_w, N,
                             t_min, t_max, p_min, p_max,
                             par_labels_theta, par_units_theta, 
                             legend_data = legend,
                             legend_fit=r'Fit: $\langle \theta(t) \rangle =  A_{\mathrm{B_z}}\cos(\omega_a t + \phi) + A_{\mathrm{EDM}}\sin(\omega_a t + \phi) + c$',
                             ylabel=r"$\langle\theta_y\rangle$ [mrad] per "+str(int(bin_w*1e3))+" ns",
                             font_size=font_size,
                             prec=2)
    cu.textL(ax, 0.74, 0.15, leg_data, fs=font_size)
    cu.textL(ax, 0.23, 0.15, leg_fit, fs=font_size, c="r")
    ax.set_xlim(0, g2period);
    ax.set_ylim(-2.9, 2.5);
    if(args.scan==False): fig.savefig("../fig/bz_truth_fit_S"+str(station)+".png", dpi=300)

    scan_label="_"+str(t_min)+"_"+str(t_max)+"_"+str(p_min)+"_"+str(p_max)+"_"+str(ndf)
    if(args.scan==True):
        par_dump=np.array([[t_min], t_max, p_min, p_max, chi2_ndf, ndf, g2period, bin_w, N, station, ds_name, *par, *par_e])
        par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "bin_w", "n",  "station", "ds"]
        par_dump_keys.extend(par_names_theta)
        par_dump_keys.extend( [str(par)+"_e" for par in par_names_theta] )
        dict_dump = dict(zip(par_dump_keys,par_dump))
        df = pd.DataFrame.from_records(dict_dump, index='start')
        with open("../DATA/scans/edm_scan_"+keys[1]+".csv", 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0)
        plt.savefig("../fig/scans/bz_truth_fit_S"+str(station)+scan_label+".png", dpi=300)

   
if __name__ == '__main__':
    main()