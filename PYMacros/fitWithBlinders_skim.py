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
from scipy import stats, optimize, fftpack
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
stations=(12, 18)
expected_DSs = ("60h", "9D", "HK", "EG")
f_a = 0.23 # MHz "g-2" frequency
f_c = 6.71 # MHz cyclotron frequency
par_names= ["N", "tau", "A", "R", "phi"]


### Get ds_name from filename
ds_name=args.hdf.replace(".","/").split("/")[-2] # if all special chars are "/" the DS name is just after extension
print("Detected DS name:", ds_name, "from the input file!")
if(not ds_name in expected_DSs):
    raise Exception("Unexpected input HDF name: if using Run-1 data, rename your file to DS.h5 (e.g. 60h.h5); Otherwise, modify functionality of this programme...exiting...")

# Now that we know what DS we have, we can
# set tune and calculate expected FFTs and
print("Setting tune parameters for ", ds_name, "DS")
if (ds_name=="60h" or ds_name=="EG"): n_tune = 0.108
if (ds_name=="9D" or ds_name=="HK"): n_tune = 0.120
f_cbo = f_c * (1 - np.sqrt(1-n_tune) )
f_vw = f_c * (1 - 2 *np.sqrt(n_tune) )
f_cbo_M_a = f_cbo - f_a
f_cbo_P_a = f_cbo + f_a

### (2) Deal with weather we are doing 5 or 9 parameter fit
p0_s12=[15000, 64.44, 0.34, -60, 2.080]
p0_s18=[13000, 64.44, 0.34, -90, 2.080]
func = cu.blinded_wiggle_function # standard blinded function from CommonUtils
func_label="5par"
legend_fit=r'Fit: $N(t)=Ne^{-t/\tau}[1+A\cos(\omega_at+\phi)]$'
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
        cbo_fit_terms_s12=[0.028, 2.332, 2.826, 175.2]
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
    p0_s12.extend([18.0]); p0_s18.extend([6.0]); # S12 and S18 
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
residuals=[[],[]]
times_binned=[[],[]]

# form a string to distinguish files (this still a list[i_station])
global_label=ds_name+"_"+func_label
file_label=["_S"+str(s)+"_"+global_label for s in stations]
scan_label="_"+str(min_t)+"_"+str(max_t)
par_e_names=[str(par)+"_e" for par in par_names]

def main():

    #open the hdf file and fit!
    times_binned, residuals=fit()

    if(args.scan==False):
        #now plot the (data - fit)
        residual_plots(times_binned, residuals)

        #FFTs
        fft(residuals)

        #finally add plots into single canvas
        canvas()

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
    s12_cut = (data['station'] == stations[0])
    s18_cut = (data['station'] == stations[1])
    station_cut = (s12_cut, s18_cut)

    for i_station, station in enumerate(stations):
        data_station=data[station_cut[i_station]]
        # resolve into t variable for ease for a station
        t = data_station['trackT0'] # already expected in us (or can just *1e-3 here if not)
        N=data_station.shape[0]
        print("Entries: ", N, " in", station)

        print("digitising data (binning)...")
        # just call x,y = frequencies, bin_centres for plotting and fitting
        x, y = cu.get_freq_bin_c_from_data( t, bin_w, (min_t, max_t) )
        y_err = np.sqrt(y) # sigma =sqrt(N)

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
                                                legend_data="Run-1: "+ds_name+" dataset S"+str(station)
                                                )
        plt.legend(fontsize=font_size-3, loc='upper center', bbox_to_anchor=(0.5, 1.1) )
        if(args.scan==False): plt.savefig("../fig/wiggle/wiggle"+file_label[i_station]+".png", dpi=300)

        # Get residuals for next set of plots
        residuals[i_station] = cu.residuals(x, y, func, par)
        times_binned[i_station] = x

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

    return times_binned, residuals

def residual_plots(times_binned, residuals):
    '''
    loop over the two lists to fill residual plots
    '''
    for i_station, (x, residual) in enumerate(zip(times_binned, residuals)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, residual, c='g', label="Run-1: "+ds_name+" dataset S"+str(stations[i_station])+" data-fit")
        ax.set_ylabel(r"Fit residuals (counts, $N$)", fontsize=font_size);
        ax.set_xlabel(r"Time [$\mathrm{\mu}$s]", fontsize=font_size)
        ax.legend(fontsize=font_size)
        if(args.scan==False): plt.savefig("../fig/res/res"+file_label[i_station]+".png", dpi=300)
        if(args.scan==True):  plt.savefig("../fig/scans/res"+file_label[i_station]+scan_label+".png", dpi=300)

def fft(residuals):
    '''
    perform the FFT analysis on the fit residuals
    '''
    print("FFT analysis...")
    for i_station, residual in enumerate(residuals):

        print("S"+str(stations[i_station]),":")
        fig, ax = plt.subplots(figsize=(8, 5))

        # de-trend data (trying to remove the peak at 0 Hz)
        res_detrend = np.subtract(residual, np.average(residuals))

        # Now to the FFT:
        N = len(res_detrend) # window length
        res_fft = fftpack.fft(res_detrend) # return DFT on the fit residuals
        res_fft = np.absolute(res_fft) # magnitude of the complex number
        freqs = fftpack.fftfreq(N, d=bin_w)  # DFT sample frequencies (d=sample spacing, ~150 ns)
        #take the +ive part
        freq=freqs[0:N//2]
        res_fft=res_fft[0:N//2]

        # Calculate the Nyquist frequency, which is twice the highest frequeny in the signal
        # or half of the sampling rate ==  the maximum frequency before sampling errors start
        sample_rate = 1.0 / bin_w
        nyquist_freq = 0.5 * sample_rate
        # print("bin width:", round(bin_w*1e3,3), " ns")
        # print("sample rate:", round(sample_rate,3), "MHz")
        # print("Nyquist freq:", round(nyquist_freq,3), "MHz\n")

        # set plot limits
        x_min, x_max, y_min, y_max = 0.0, nyquist_freq, 0,  1.2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ###Normalise and plot:
        # find index of frequency above x_min
        index=next(i for i,v in enumerate(freq) if v > x_min)
        # arbitrary: scale by max value in range of largest non-zero peak
        norm = 1./max(res_fft[index:-1])
        if(args.loss): norm=norm*0.1 # scale by 4 if LM is used
        res_fft=res_fft*norm
        ax.plot(freq, res_fft, label="Run-1: "+ds_name+" dataset S"+str(stations[i_station])+r": FFT, $n$={0:.3f}".format(n_tune), lw=2, c="g")

        #plot expected frequencies
        ax.plot( (f_cbo, f_cbo), (y_min, y_max), c="r", ls="--", label="CBO")
        ax.plot( (f_a, f_a), (y_min, y_max), c="b", ls="-", label=r"$(g-2)$")
        ax.plot( (f_cbo_M_a, f_cbo_M_a), (y_min, y_max), c="k", ls="-.", label=r"CBO - $(g-2)$")
        ax.plot( (f_cbo_P_a, f_cbo_P_a), (y_min, y_max), c="c", ls=":", label=r"CBO + $(g-2)$")
        ax.plot( (f_vw, f_vw), (y_min, y_max), c="m", ls=(0, (1, 10)), label="VW")

        # prettify and save plot
        ax.legend(fontsize=font_size, loc="best")
        ax.set_ylabel("FFT magnitude (normalised)", fontsize=font_size)
        ax.set_xlabel("Frequency [MHz]", fontsize=font_size)
        if(args.scan==False): plt.savefig("../fig/fft/fft"+file_label[i_station]+".png", dpi=300)
        if(args.scan==True):  plt.savefig("../fig/scans/fft"+file_label[i_station]+scan_label+".png", dpi=300)


def canvas():
    if (args.scan==False):
        subprocess.call(["convert" , "+append", "../fig/wiggle/wiggle"+file_label[0]+".png" , "../fig/wiggle/wiggle"+file_label[1]+".png", "../fig/wiggle/wiggle"+global_label+".png"])
        subprocess.call(["convert" , "+append", "../fig/fft/fft"+file_label[0]+".png" , "../fig/fft/fft"+file_label[1]+".png", "../fig/fft/fft"+global_label+".png"])
        subprocess.call(["convert" , "-append", "../fig/wiggle/wiggle"+global_label+".png" , "../fig/fft/fft"+global_label+".png", "../fig/"+global_label+".png"])
        print("Final plot: ", "../fig/"+global_label+".png")

if __name__ == "__main__":
    main()
