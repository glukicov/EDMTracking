# Author: Gleb Lukicov (21 Feb 2020)
# Perform a 5-parameter blinded fit 
# on skimmed data in HDF5 format (from skimTrees.py module)

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
arg_parser.add_argument("--p0", nargs='+', type=float, default=[85263, 64.0, 0.33, 1.0, 2.0]) #fit parameters (initial guess)
arg_parser.add_argument("--max", type=float, default=400.0) #max fit time 
arg_parser.add_argument("--min", type=float, default=30.0) #max fit time 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/MMA/60h.h5") #input data 
arg_parser.add_argument("--key", type=str, default="QualityTracks") # or QualityVertices
arg_parser.add_argument("--label", type=str, default="60h") # or QualityVertices
args=arg_parser.parse_args()

### Constants 
stations=(12, 18)

#define modulation and limits (more can be made as arguments) 
bin_w = 150*1e-3 # 150 ns 
min_x = args.min # us
max_x = args.max # us 
t_mod=100 # us; fold plot every N us 

# expected frequencies in FFTs 
cbo_f = 0.37 # MHz n=0.108 (60h)
gm2_f = 0.23 # MHz 
cbo_M_gm2_f = cbo_f - gm2_f 
cbo_P_gm2_f = cbo_f + gm2_f
vm_f = 10*gm2_f  #  (60h) analyical: f_vm = f_c - 2*f_ybo

### Global storage 
residuals=[[],[]] 
times_binned=[[],[]] 

def main():

    #open the hdf file and fit! 
    times_binned, residuals=fit(scan=args.scan)

    #now plot the (data - fit)
    residual_plots(times_binned, residuals)

    #FFTs
    fft(residuals, scan=args.scan)

    #finally add plots into single canvas
    canvas()

def fit(scan=False):
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
    print("Found", data.shape[0], "entries\n")
    
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
        x, y = cu.get_freq_bin_c_from_data( t, bin_w, (min_x, max_x) )
        y_err = np.sqrt(y) # sigma =sqrt(N)
        print("digitised data done!")

        print("Fitting...")
        # Levenberg-Marquardt algorithm as implemented in MINPACK
        par, pcov = optimize.curve_fit(f=cu.blinded_wiggle_function, xdata=x, ydata=y, sigma=y_err, p0=args.p0, absolute_sigma=False, method='lm')
        par_e = np.sqrt(np.diag(pcov))
        print("Pars  :", np.array(par))
        print("Pars e:",np.array(par_e))
        chi2_ndf, chi2, ndf=cu.chi2_ndf(x, y, y_err, cu.blinded_wiggle_function, par)
        print("Fit 𝝌2/DoF="+str(round(chi2_ndf,1)) )

        print("Plotting fit and data!\n")
        #make more automated things for "plot prettiness"
        data_type = re.findall('[A-Z][^A-Z]*', args.key) # should break into "Quality" and "Tracks"/"Vertices"

        #use pre-define module wiggle function from CommonUtils
        fig,ax = cu.modulo_wiggle_fit_plot(x, y, t_mod, max_x, min_x, N, par, par_e, chi2_ndf, bin_w,
                                                prec=3,
                                                key=data_type[0]+" "+data_type[1], 
                                                legend_data="Run-1: "+args.label+" dataset S"+str(station) 
                                                )
        plt.legend(fontsize=font_size-3, loc='upper center', bbox_to_anchor=(0.5, 1.1) )
        plt.savefig("../fig/wiggle_S"+str(station)+"_"+args.label+".png", dpi=300)
        
        # Get residuals for next set of plots  
        residuals[i_station] = cu.residuals(x, y, cu.blinded_wiggle_function, par)
        times_binned[i_station] = x

        # if running externally, via a different module and passing scan==True as an argument
        # dump the parameters to a unique file for summary plots 
        if(scan==True):
            par_dump=np.array(chi2_ndf, par) 
            file_label=args.label+"_S"+str(station)+"_"+str(args.min)+"_"+str(args.max)
            np.save("../DATA/misc/scans/data_"+file_label+".npy", par_dump)
            plt.savefig("../DATA/misc/scans/wiggle_"+file_label+".png", dpi=300) 
    
    return times_binned, residuals

def residual_plots(times_binned, residuals):
    '''
    loop over the two lists to fill residual plots
    '''
    for i_station, (x, residual) in enumerate(zip(times_binned, residuals)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, residual, c='g', label="Run-1: "+args.label+" dataset S"+str(stations[i_station])+" data-fit")
        ax.set_ylabel(r"Fit residuals (counts, $N$)", fontsize=font_size);
        ax.set_xlabel(r"Time [$\mathrm{\mu}$s]", fontsize=font_size)
        ax.legend(fontsize=font_size)
        plt.savefig("../fig/res_S"+str(stations[i_station])+"_"+args.label+".png", dpi=300)


def fft(residuals, scan=False):
    '''
    perform the FFT analysis on the fit residuals 
    '''
    print("FFT analysis...")
    for i_station, residual in enumerate(residuals):
        
        print("S"+str(stations[i_station]),":")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # detrend data (trying to remove the peak at 0 Hza)
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
        print("bin width:", round(bin_w*1e3,3), " ns")
        print("sample rate:", round(sample_rate,3), "MHz")
        print("Nyquist freq:", nyquist_freq, "MHz\n")

        # set plot limits
        x_min, x_max, y_min, y_max = 0.02, nyquist_freq, 0,  1.2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
        ###Normalise and plot:
        # find index of frequency above x_min 
        index=next(i for i,v in enumerate(freq) if v > x_min) 
        # arbitrary: scale by max value in range of largest non-zero peak
        norm = 1./max(res_fft[index:-1])
        res_fft=res_fft*norm
        ax.plot(freq, res_fft, label="Run-1: "+args.label+" dataset S"+str(stations[i_station])+": FFT", lw=2, c="g")
       
        #plot expected frequencies
        ax.plot( (cbo_f, cbo_f), (y_min, y_max), c="r", ls="--", label="CBO")
        ax.plot( (gm2_f, gm2_f), (y_min, y_max), c="b", ls="-", label=r"$(g-2)$")
        ax.plot( (cbo_M_gm2_f, cbo_M_gm2_f), (y_min, y_max), c="k", ls="-.", label=r"CBO - $(g-2)$")
        ax.plot( (cbo_P_gm2_f, cbo_P_gm2_f), (y_min, y_max), c="c", ls=":", label=r"CBO + $(g-2)$")
        ax.plot( (vm_f, vm_f), (y_min, y_max), c="m", ls=(0, (1, 10)), label="VW")
        
        # prettify and save plot 
        ax.legend(fontsize=font_size, loc="best")
        ax.set_ylabel("FFT magnitude (normalised)", fontsize=font_size)
        ax.set_xlabel("Frequency [MHz]", fontsize=font_size)
        plt.savefig("../fig/fft_S"+str(stations[i_station])+"_"+args.label+".png", dpi=300)


def canvas():
    subprocess.call(["convert" , "+append", "../fig/wiggle_S"+str(stations[0])+"_"+args.label+".png" , "../fig/wiggle_S"+str(stations[1])+"_"+args.label+".png", "../fig/wiggle_"+args.label+".png"])
    subprocess.call(["convert" , "+append", "../fig/fft_S"+str(stations[0])+"_"+args.label+".png" , "../fig/fft_S"+str(stations[1])+"_"+args.label+".png", "../fig/fft_"+args.label+".png"])
    subprocess.call(["convert" , "-append", "../fig/wiggle_"+args.label+".png" , "../fig/fft_"+args.label+".png", "../fig/"+args.label+".png"])
    print("Final plot: ", "../fig/"+args.label+".png")

if __name__ == "__main__":
    main()