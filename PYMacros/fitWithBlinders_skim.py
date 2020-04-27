# Author: Gleb Lukicov (21 Feb 2020)
# Perform a 5 or 9-parameter blinded fit
# on skimmed data in HDF5 format (from skimTrees.py module)

# For Run-1 data DS names, labels, tune and expected FFTs are defined for the four DSs (e.g. 60h.h5 == 60h)
# For Run-2 or unexpected fileName.h5 this needs to be extended...

#Blinding lib imported in CommonUtils:
import sys, re, subprocess
import argparse
import pandas as pd
import numpy as np
np.set_printoptions(precision=3) # 3 sig.fig
sys.path.append('../CommonUtils/') # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
from scipy import stats, optimize
import matplotlib as mpl
mpl.use('Agg') # MPL in batch mode
font_size=15
import matplotlib.pyplot as plt

#Input fitting parameters
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/MMA/60h.h5", help="input data")
arg_parser.add_argument("--key", type=str, default="QualityTracks", help="or QualityVertices")
arg_parser.add_argument("--min", type=float, default=30.2876, help="min fit starts time")
arg_parser.add_argument("--max", type=float, default=450.0, help="max fit start time")
arg_parser.add_argument("--cbo", action='store_true', default=False, help="include extra 4 CBO terms if True in the fitting")
arg_parser.add_argument("--loss", action='store_true', default=False, help="include extra kloss")
arg_parser.add_argument("--scan", action='store_true', default=False, help="if run externally for iterative scans - dump 𝝌2 and fitted pars to a file for summary plots")
arg_parser.add_argument("--corr", action='store_true', default=False, help="Save covariance matrix for plotting")
args=arg_parser.parse_args()

if(args.loss==True): args.cbo = True 

### Constants
# stations=(12, 18)
stations=([1218])
expected_DSs = ("60h", "9D", "HK", "EG", "Sim", "R1")
official_DSs = ("1a", "1c", "1b", "1e", "Sim", "1")
par_names= ["N", "tau", "A", "R", "phi"]

### Get ds_name from filename
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
ds_name_official=official_DSs[expected_DSs.index(ds_name)]
print("Detected DS name:", ds_name, ds_name_official, "from the input file!")
if(not ds_name in expected_DSs):
    raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")

# Now that we know what DS we have, we can
# set tune and calculate expected FFTs and
if (ds_name == "R1"): ds_name = "EG"
cu._DS=ds_name
print("Setting tune parameters for", ds_name, "DS")

### (2) Deal with weather we are doing 5 or 9 parameter fit
p0_s12=[30000, 63, 0.3395, -56, 2.072]
p0_s18=[13000, 64.44, 0.34, -90, 2.080]
func = cu.blinded_wiggle_function # standard blinded function from CommonUtils
func_label="5par"
legend_fit=r'Fit: $N(t)=N_{0}e^{-t/\tau}[1+A\cos(\omega_at+\phi)]$'
show_cbo_terms=False
show_loss_terms=False

# for CBO pars see Joe's DocDB:12933
if (args.cbo):
    if (ds_name=="HK"): 
        cbo_fit_terms_s12=[0.05, 2.5, 3.8, 90.0]   
        cbo_fit_terms_s18=[0.05, 2.5, 3.8, 90.0]   
        print("Bad resistors in HK are accounted in the starting parameters!") # bad resistors fix 
    if (ds_name=="EG"): 
        cbo_fit_terms_s12=[-0.033, 2.312, 5.233, 100.0]
        cbo_fit_terms_s18=[0.0051, 2.300, 1.625, 120.0]
        print("Bad resistors in EG are accounted in the starting parameters!") # bad resistors fix 
    if (ds_name=="60h"):  
        # cbo_fit_terms_s12=[0.05, 2.5, 3.8, 90.0] 
        cbo_fit_terms_s12=[0.022, 2.330, 2.4, 149]
        cbo_fit_terms_s18=[0.020, 2.338, 1.055, 175.2]
    if (ds_name=="9D"):  
        # cbo_fit_terms_s12=[0.026, 2.330, 3.034, 200.0]
        cbo_fit_terms_s12=[0.05, 2.5, 3.8, 90.0]  
        cbo_fit_terms_s18=[0.034, 2.327, 1.774, 120.2]
    p0_s12.extend(cbo_fit_terms_s12)    
    p0_s18.extend(cbo_fit_terms_s18)    
    par_names.extend(["A_cbo", "w_cbo", "phi_cbo", "tau_cbo"])
    func=cu.blinded_wiggle_function_cbo
    func_label="9par"
    legend_fit=legend_fit+r"$\cdot C(t)$"
    show_cbo_terms=True

if(args.loss):
    cu._DS=ds_name # for muon loss integral 
    par_names.extend(["K_LM"])
    p0_s12.extend([6.0]); p0_s18.extend([6.0]); # S12 and S18 
    func=cu.blinded_10_par
    func_label="10par"
    legend_fit=legend_fit+r"$\cdot\Lambda(t)$"
    show_cbo_terms=True
    show_loss_terms=True

#form final starting parameters array (should probably used dict for this...)
p0=[p0_s12, p0_s18]

#define modulation and limits (more can be made as arguments)
bin_w = 149.2*1e-3 # 150 ns
min_t = args.min # us
max_t = args.max # us
t_mod=100 # us; fold plot every N us

### Global storage
residuals=[]
errors=[]
times_binned=[]


# form a string to distinguish files (this still a list[i_station])
global_label=ds_name+"_"+func_label
file_label=["_S"+str(s)+"_"+global_label for s in stations]
scan_label="_"+str(min_t)+"_"+str(max_t)
par_e_names=[str(par)+"_e" for par in par_names]

def main():

    #open the hdf file and fit!
    times_binned, residuals, errors =fit()

    if(args.scan==False):
        #now plot the (data - fit)
        # cu.residual_plots(times_binned, residuals, file_label=file_label, scan_label=scan_label)
        

        # print(len(residuals))
        # print(len(errors))
        # print(file_label)
        # sys.exit()

        cu.pull_plots(residuals, errors, file_label=file_label)

        #FFTs
        cu.fft(residuals, bin_w, file_label=file_label, scan_label=scan_label)

        #finally add plots into single canvas
        #canvas()

def fit():
    '''
    Open HDF5, chose the station
    apply further time cuts
    take-in initial fitting parameters
    do the fit,plot as the modulated wiggle
    get 𝝌2, pass residuals as the output

    if scan==True dump 𝝌2 etc. into a file
    '''
    print("Opening", args.key, "in", args.hdf, "...")
    data = pd.read_hdf(args.hdf, args.key)
    print("Found", data.shape[0], "entries")
    time_cut = ( (data['trackT0'] > min_t) & (data['trackT0'] < max_t) )
    data = data[time_cut]
    print("After time cut:", data.shape[0], "entries")
    print("Fitting from", min_t, "to", max_t,"[μs] using", func_label)

    #define station cuts to loop over
    if (len(stations)==2):
        s12_cut = (data['station'] == stations[0])
        s18_cut = (data['station'] == stations[1])
        station_cut = (s12_cut, s18_cut)

    for i_station, station in enumerate(stations):
        if (len(stations)==2):  data_station=data[station_cut[i_station]]
        if (len(stations)==1): data_station=data
        
        # resolve into t variable for ease for a station
        t = data_station['trackT0'] # already expected in us (or can just *1e-3 here if not)
        N=data_station.shape[0]
        print("Entries: ", N, " in", station)

        print("digitising data (binning)...")
        # just call x,y = frequencies, bin_centres for plotting and fitting
        x, y, y_err = cu.get_freq_bin_c_from_data( t, bin_w, (min_t, max_t) )

        print("Fitting...")
        par, par_e, pcov, chi2_ndf, ndf =cu.fit_and_chi2(x, y, y_err, func, p0[i_station])       
        if (np.max(abs(par_e)) == np.Infinity ): raise Exception("\nOne of the fit parameters is infinity! Exiting...\n")
        if(args.corr): print("Covariance matrix", pcov); np.save("../DATA/misc/pcov_S"+str(station)+".np", pcov);

        print("Plotting fit and data!")
        #make more automated things for "plot prettiness"
        data_type = re.findall('[A-Z][^A-Z]*', args.key) # should break into "Quality" and "Tracks"/"Vertices"

        #use pre-define module wiggle function from CommonUtils
        fig,ax = cu.modulo_wiggle_fit_plot(x, y, func, par, par_e, chi2_ndf, ndf, t_mod, max_t, min_t, bin_w,  N,
                                                show_cbo_terms=show_cbo_terms,
                                                show_loss_terms=show_loss_terms,
                                                legend_fit=legend_fit,
                                                prec=3,
                                                key=data_type[0]+" "+data_type[1],
                                                legend_data="Run-"+ds_name_official+" dataset S"+str(station)
                                                )
        plt.legend(fontsize=font_size-3, loc='upper center', bbox_to_anchor=(0.5, 1.0) )
        if(args.scan==False): plt.savefig("../fig/wiggle/wiggle"+file_label[i_station]+".png", dpi=200)

        # Get residuals for next set of plots
        residuals = cu.residuals(x, y, func, par)
        times_binned = x
        errors=y_err

        # if running externally, via a different module and passing scan==True as an argument
        # dump the parameters to a unique file for summary plots
        if(args.scan==True):
            par_dump=np.array([[min_t], max_t, chi2_ndf, ndf, N, station, ds_name, *par, *par_e])
            par_dump_keys = ["start", "stop", "chi2", "ndf", "n", "station", "ds"]
            par_dump_keys.extend(par_names)
            par_dump_keys.extend(par_e_names)
            dict_dump = dict(zip(par_dump_keys,par_dump))
            df = pd.DataFrame.from_records(dict_dump, index='start')
            with open("../DATA/scans/scan.csv", 'a') as f:
                df.to_csv(f, mode='a', header=f.tell()==0)
            plt.savefig("../fig/scans/wiggle"+file_label[i_station]+scan_label+".png", dpi=300)

    return times_binned, residuals, errors


def canvas():
    if (args.scan==False):
        subprocess.call(["convert" , "+append", "../fig/wiggle/wiggle"+file_label[0]+".png" , "../fig/wiggle/wiggle"+file_label[1]+".png", "../fig/wiggle/wiggle"+global_label+".png"])
        subprocess.call(["convert" , "+append", "../fig/fft/fft"+file_label[0]+".png" , "../fig/fft/fft"+file_label[1]+".png", "../fig/fft/fft"+global_label+".png"])
        subprocess.call(["convert" , "-append", "../fig/wiggle/wiggle"+global_label+".png" , "../fig/fft/fft"+global_label+".png", "../fig/"+global_label+".png"])
        print("Final plot: ", "../fig/"+global_label+".png")

if __name__ == "__main__":
    main()
