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
# arg_parser.add_argument("--t_min", type=float, default=4.3) # us 
arg_parser.add_argument("--t_min", type=float, default=30.56) # us 
arg_parser.add_argument("--t_max", type=float, default=454.00) # us 
arg_parser.add_argument("--p_min", type=float, default=1800) # us 
arg_parser.add_argument("--p_max", type=float, default=3100) # us 
arg_parser.add_argument("--bin_w", type=float, default=15) # ns 
arg_parser.add_argument("--g2period", type=float, default=None) # us 
arg_parser.add_argument("--phase", type=float, default=None) # us 
arg_parser.add_argument("--lt", type=float, default=None) # us 
# arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/Sim/Sim.h5") 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/EDM/60h.h5", help="input data")
arg_parser.add_argument("--corr", action='store_true', default=False, help="Save covariance matrix for plotting")
arg_parser.add_argument("--scan", action='store_true', default=False, help="if run externally for iterative scans - dump 𝝌2 and fitted pars to a file for summary plots") 
arg_parser.add_argument("--count", action='store_true', default=False)
arg_parser.add_argument("--hist", action='store_true', default=False)
args=arg_parser.parse_args()

### Define constants and starting fit parameters
font_size=14 # for plots
stations=(12, 18)
expected_DSs = ("60h", "9D", "HK", "EG", "Sim")

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
par_names_count= ["N", "tau", "A", "phi"]; par_labels_count= [r"$N$", r"$\tau$", r"$A$", r"$\phi$"]; par_units_count=[" ",  r"$\rm{\mu}$s", " ", "rad"]
par_names_theta= ["A_Bz", "A_edm_blind", "c"]; par_labels_theta= [r"$A_{B_{z}}$", r"$A^{\rm{BLINDED}}_{\mathrm{EDM}}$", r"$c$"]; par_units_theta=[r"$\rm{\mu}$rad", r"$\rm{\mu}$rad", r"$\rm{\mu}$rad"]
par_names_theta_truth=par_names_theta.copy(); par_names_theta_truth[1]="A_edm"; par_labels_truth=par_labels_theta.copy(); par_labels_truth[1]=r"$A_{\mathrm{EDM}}$"
p0_count=[ [15000, 63.4, 0.339, 2.057], [12000, 64.4, 0.341, 2.074]]
p0_theta_blinded=[ [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
if(sim): 
    p0_count=[ [3000, 64.4, -0.40, 6.240], [3000, 64.4, -0.40, 6.240]]
    p0_theta_blinded=[ [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
print("Starting pars theta blinded", *par_names_theta, *p0_theta_blinded)
print("Starting pars count",*par_names_count, *p0_count)
p0_theta_truth=[ [0.00, 0.17, 0.0], [0.00, 0.17, 0.0] ]; print("Starting pars TRUTH theta", *par_names_theta_truth, *p0_theta_truth)

### Define global variables
residuals_counts, residuals_theta, times_counts, times_theta=[[],[]], [[],[]], [[],[]] , [[],[]]
if(sim): residuals_counts, residuals_theta, times_counts, times_theta=[ [] ], [ [] ], [ [] ], [ [] ]

# output scan file HDF5 keys 
keys=["count", "theta", "truth"]
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

        #############
        #Plot counts vs. mod time and fit
        #############
        ### Digitise data
        x, y, y_e = cu.get_freq_bin_c_from_data(data_station['mod_times'], bin_w, (0, g2period) )
       
        #Fit
        par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.unblinded_wiggle_fixed, p0_count[i_station])
        if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
        if (args.corr): print("Covariance matrix\n", pcov); np.save("../DATA/misc/pcov_count_S"+str(station)+".np", pcov);

        #Plot
        if(sim): legend=ds_name+" S"+str(station);
        else:    legend="Run-1: "+ds_name+" dataset S"+str(station);                           
        fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.unblinded_wiggle_fixed, 
                                     par, par_e, chi2_ndf, ndf, bin_w, N,
                                     t_min, t_max, p_min, p_max,
                                     par_labels_count, par_units_count, 
                                     legend_data = legend,
                                     legend_fit=r'Fit: $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]$',
                                     ylabel=r"Counts ($N$) per "+str(int(bin_w*1e3))+" ns",
                                     font_size=font_size,
                                     prec=3)
        
        if(sim): cu.textL(ax, 0.48, 0.35, leg_fit, c="r", fs=font_size+1); cu.textL(ax, 0.80, 0.75, leg_data, fs=font_size+1)
        if(not sim): cu.textL(ax, 0.65, 0.30, leg_fit, c="r", fs=font_size+1); cu.textL(ax, 0.23, 0.6, leg_data, fs=font_size+1)
        ax.set_ylim(np.amin(y)*0.9, np.amax(y)*1.15);
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
    
        # get residuals for later plots 
        residuals_counts[i_station] = cu.residuals(x, y, cu.unblinded_wiggle_fixed, par)
        times_counts[i_station] = x

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

        # cu._LT=64.44
        # print("LT set to", round(cu._LT,2), "us")
        # cu._phi=5.3
        # cu._phi=2.06
        # print("Phase set to", round(cu._phi,2), "rad")

        if(not args.count):

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
            if(args.corr): print("Covariance matrix\n", pcov); np.save("../DATA/misc/pcov_theta_S"+str(station)+".np", pcov);

            #Plot
            fig, ax, leg_data, leg_fit = cu.plot_edm(x, y, y_e, cu.thetaY_phase, 
                                         par, par_e, chi2_ndf, ndf, bin_w, N,
                                         t_min, t_max, p_min, p_max,
                                         par_labels_theta, par_units_theta, 
                                         legend_data = legend,
                                         legend_fit=r'Fit: $\langle \theta(t) \rangle =  A_{\mathrm{B_z}}\cos(\omega_a t + \phi) + A_{\mathrm{EDM}}\sin(\omega_a t + \phi) + c$',
                                         ylabel=r"$\langle\theta_y\rangle$ [mrad] per "+str(int(bin_w*1e3))+" ns",
                                         font_size=font_size,
                                         prec=2, urad=True)
            ax.set_xlim(0, g2period);
            ax.set_ylim(ax.get_ylim()[0]*1.6, ax.get_ylim()[1]*1.2);
            if(not sim): ax.set_ylim(ax.get_ylim()[0]*1.23, ax.get_ylim()[1]*1.4)
            cu.textL(ax, 0.75, 0.15, leg_data, fs=font_size)
            cu.textL(ax, 0.25, 0.17, leg_fit, fs=font_size, c="r")
            print("Fit in "+ds_name+" S:"+str(station), leg_fit)
            if(args.scan==False): fig.savefig("../fig/bz_"+ds_name+"_S"+str(station)+".png", dpi=300)

            if(args.scan==True):
                par_dump=np.array([[t_min], t_max, p_min, p_max, chi2_ndf, ndf, g2period, cu._LT, cu._phi,  bin_w, bin_n, xy_bins[0], xy_bins[1], N, station, ds_name, *par, *par_e])
                par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "lt", "phase", "bin_w", "bin_n", "ndf_x", "ndf_y", "n", "station", "ds"]
                par_dump_keys.extend(par_names_theta)
                par_dump_keys.extend( [str(par)+"_e" for par in par_names_theta] )
                dict_dump = dict(zip(par_dump_keys,par_dump))
                df = pd.DataFrame.from_records(dict_dump, index='start')
                with open("../DATA/scans/edm_scan_"+keys[1]+".csv", 'a') as f:
                    df.to_csv(f, mode='a', header=f.tell()==0)
                plt.savefig("../fig/scans/bz_"+ds_name+"_S"+str(station)+scan_label+".png", dpi=300)


            # get residuals for later plots 
            residuals_theta[i_station] = cu.residuals(x, y, cu.thetaY_phase, par)
            times_theta[i_station] = x


        #make sanity plots 
        if(args.hist):
        # if(not sim and args.hist):
            
            fig, _ = plt.subplots()
            n_bins_ang=400*5
            ax, _ = cu.plotHist(ang, n_bins=n_bins_ang, prec=3, units="mrad", label=ds_name+" dataset S"+str(station) )
            legend=cu.legend3_sd(*cu.stats3_sd(ang), "mrad", prec=3)
            cu.textL(ax, 0.8, 0.85, str(legend), fs=14)
            ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.1)
            ax.set_xlim(-50,50)
            ax.set_xlabel(r"$\theta_y$ [mrad]", fontsize=font_size);
            ax.set_ylabel("Entries per "+str(round((max(ang)-min(ang))/n_bins_ang,3))+" mrad", fontsize=font_size);
            ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.3, 1.0))
            fig.savefig("../fig/theta_"+ds_name+"_S"+str(station)+".png", dpi=300, bbox_inches='tight')

            fig, _ = plt.subplots()
            n_binsXY_ang=(575,575)
            jg, cb, legendX, legendY = cu.plotHist2D(data_station['trackT0'], ang, n_binsXY=n_binsXY_ang, prec=2, unitsXY=(r"[$\rm{\mu}$s]", "mrad"), label="S"+str(station), cmin=5)
            jg.ax_joint.set_xlim(30.6, 140)
            jg.ax_joint.set_ylim(-50, 50)
            jg.ax_joint.set_ylabel(r"$\theta_y$ [mrad]", fontsize=font_size+2);
            jg.ax_joint.set_xlabel(r"t [$\rm{\mu}$s]", fontsize=font_size+2);
            plt.savefig("../fig/theta2D_"+ds_name+"_S"+str(station)+".png", dpi=300, bbox_inches='tight')

            fig, _ = plt.subplots()
            jg, cb, legendX, legendY = cu.plotHist2D(data_station['mod_times'], ang, n_binsXY=n_binsXY_ang, prec=3, unitsXY=(r"[$\rm{\mu}$s]", "mrad"), label="S"+str(station), cmin=5 )
            jg.ax_joint.set_xlim(0.0, g2period)
            jg.ax_joint.set_ylim(-50, 50)
            jg.ax_joint.set_ylabel(r"$\theta_y$ [mrad]", fontsize=font_size+2);
            jg.ax_joint.set_xlabel(r"$t^{mod}_{g-2}$"+r"[$\rm{\mu}$s]", fontsize=font_size+2);
            plt.savefig("../fig/theta2D_mod_"+ds_name+"_S"+str(station)+".png", dpi=300, bbox_inches='tight')
            
        #############
        # Make truth (un-blinded fits) if simulation
        #############
        if(sim):
            print("Making truth plots in simulation")

            # Bin 
            df_binned =cu.Profile(data_station['mod_times'], data_station['theta_y_mrad'], None, nbins=bin_n, xmin=np.min(data_station['mod_times']), xmax=np.max(data_station['mod_times']), mean=True, only_binned=True)
            x, y, y_e, x_e =df_binned['bincenters'], df_binned['ymean'], df_binned['yerr'], df_binned['xerr']

            # Fit 
            par, par_e, pcov, chi2_ndf, ndf = cu.fit_and_chi2(x, y, y_e, cu.thetaY_phase, p0_theta_truth[i_station])
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
            cu.textL(ax, 0.74, 0.15, leg_data, fs=font_size)
            cu.textL(ax, 0.23, 0.12, leg_fit, fs=font_size, c="r")
            ax.set_xlim(0, g2period);
            ax.set_ylim(-np.amax(y)*1.8, np.amax(y)*2.1);
            if(not sim): ax.set_ylim(-np.amax(y)*2.5, np.amax(y)*2.5);
            if(args.scan==False): fig.savefig("../fig/bz_truth_fit_S"+str(station)+".png", dpi=300)

            if(args.scan==True):
                par_dump=np.array([[t_min], t_max, p_min, p_max, chi2_ndf, ndf, g2period, bin_w, N, station, ds_name, *par, *par_e])
                par_dump_keys = ["start", "stop", "p_min", "p_max", "chi2", "ndf", "g2period", "bin_w", "n",  "station", "ds"]
                par_dump_keys.extend(par_names_theta_truth)
                par_dump_keys.extend( [str(par)+"_e" for par in par_names_theta_truth] )
                dict_dump = dict(zip(par_dump_keys,par_dump))
                df = pd.DataFrame.from_records(dict_dump, index='start')
                with open("../DATA/scans/edm_scan_"+keys[2]+".csv", 'a') as f:
                    df.to_csv(f, mode='a', header=f.tell()==0)
                plt.savefig("../fig/scans/bz_truth_fit_S"+str(station)+scan_label+".png", dpi=300)

        #-------end of looping over stations

    ### now if not scanning - get FFTs for both stations
    # ## FFTs
    # if(not args.scan):
    #     if (not args.count):
    #         print("Plotting residuals and FFTs...")
    #         cu.residual_plots(times_counts, residuals_counts, sim=sim, eL="count", file_label=file_label)
    #         cu.fft(residuals_counts, bin_w, sim=sim, eL="count", file_label=file_label)
    #         cu.residual_plots(times_theta, residuals_theta, sim=sim, eL="theta",  file_label=file_label)
    #         cu.fft(residuals_theta, bin_w, sim=sim, eL="theta", file_label=file_label)

if __name__ == '__main__':
    main()